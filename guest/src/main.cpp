#include "imb_protocol.h"
#include "imb_wire.hpp"

#include <algorithm>
#include <cerrno>
#include <csignal>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <sys/socket.h>
#include <linux/vm_sockets.h>
#endif

namespace {

class Transport {
public:
    virtual ~Transport() = default;
    virtual void writeAll(const std::uint8_t* data, std::size_t size) = 0;
    virtual void readAll(std::uint8_t* data, std::size_t size) = 0;
};

class FileDescriptorTransport final : public Transport {
public:
    FileDescriptorTransport(int readFD, int writeFD, bool ownsFDs)
        : readFD_(readFD), writeFD_(writeFD), ownsFDs_(ownsFDs) {}

    ~FileDescriptorTransport() override {
        if (ownsFDs_) {
            if (writeFD_ >= 0) ::close(writeFD_);
            if (readFD_ >= 0 && readFD_ != writeFD_) ::close(readFD_);
        }
    }

    void writeAll(const std::uint8_t* data, std::size_t size) override {
        std::size_t offset = 0;
        while (offset < size) {
            const ssize_t written = ::write(writeFD_, data + offset, size - offset);
            if (written < 0 && errno == EINTR) continue;
            if (written <= 0) throw std::runtime_error("transport write failed: " + std::string(std::strerror(errno)));
            offset += static_cast<std::size_t>(written);
        }
    }

    void readAll(std::uint8_t* data, std::size_t size) override {
        std::size_t offset = 0;
        while (offset < size) {
            const ssize_t count = ::read(readFD_, data + offset, size - offset);
            if (count < 0 && errno == EINTR) continue;
            if (count < 0) throw std::runtime_error("transport read failed: " + std::string(std::strerror(errno)));
            if (count == 0) throw std::runtime_error("host closed the protocol stream");
            offset += static_cast<std::size_t>(count);
        }
    }

protected:
    int readFD_;
    int writeFD_;
    bool ownsFDs_;
};

#if defined(__linux__)
class VsockListenerTransport final : public Transport {
public:
    explicit VsockListenerTransport(std::uint32_t port) : connectionFD_(acceptConnection(port)) {}

    ~VsockListenerTransport() override {
        if (connectionFD_ >= 0) ::close(connectionFD_);
    }

    void writeAll(const std::uint8_t* data, std::size_t size) override {
        std::size_t offset = 0;
        while (offset < size) {
            const ssize_t written = ::write(connectionFD_, data + offset, size - offset);
            if (written < 0 && errno == EINTR) continue;
            if (written <= 0) throw std::runtime_error("vsock write failed: " + std::string(std::strerror(errno)));
            offset += static_cast<std::size_t>(written);
        }
    }

    void readAll(std::uint8_t* data, std::size_t size) override {
        std::size_t offset = 0;
        while (offset < size) {
            const ssize_t count = ::read(connectionFD_, data + offset, size - offset);
            if (count < 0 && errno == EINTR) continue;
            if (count < 0) throw std::runtime_error("vsock read failed: " + std::string(std::strerror(errno)));
            if (count == 0) throw std::runtime_error("host closed the vsock protocol stream");
            offset += static_cast<std::size_t>(count);
        }
    }

private:
    static int acceptConnection(std::uint32_t port) {
        const int listener = ::socket(AF_VSOCK, SOCK_STREAM | SOCK_CLOEXEC, 0);
        if (listener < 0) {
            throw std::runtime_error("vsock socket failed: " + std::string(std::strerror(errno)));
        }

        sockaddr_vm address{};
        address.svm_family = AF_VSOCK;
        address.svm_port = port;
        address.svm_cid = VMADDR_CID_ANY;

        if (::bind(listener, reinterpret_cast<const sockaddr*>(&address), sizeof(address)) != 0) {
            const std::string message = "vsock bind failed: " + std::string(std::strerror(errno));
            ::close(listener);
            throw std::runtime_error(message);
        }
        if (::listen(listener, 1) != 0) {
            const std::string message = "vsock listen failed: " + std::string(std::strerror(errno));
            ::close(listener);
            throw std::runtime_error(message);
        }

        std::cerr << "imb-guest-probe: listening on vsock port " << port << '\n';
        int connection = -1;
        do {
            connection = ::accept(listener, nullptr, nullptr);
        } while (connection < 0 && errno == EINTR);
        const int savedErrno = errno;
        ::close(listener);
        if (connection < 0) {
            throw std::runtime_error("vsock accept failed: " + std::string(std::strerror(savedErrno)));
        }
        std::cerr << "imb-guest-probe: host vsock connected\n";
        return connection;
    }

    int connectionFD_ = -1;
};
#endif

class ChildProcessTransport final : public Transport {
public:
    explicit ChildProcessTransport(const std::string& executable) {
        int toChild[2];
        int fromChild[2];
        if (::pipe(toChild) != 0) throw std::runtime_error("pipe failed");
        if (::pipe(fromChild) != 0) {
            ::close(toChild[0]);
            ::close(toChild[1]);
            throw std::runtime_error("pipe failed");
        }

        child_ = ::fork();
        if (child_ < 0) {
            closePipe(toChild);
            closePipe(fromChild);
            throw std::runtime_error("fork failed");
        }
        if (child_ == 0) {
            ::dup2(toChild[0], STDIN_FILENO);
            ::dup2(fromChild[1], STDOUT_FILENO);
            closePipe(toChild);
            closePipe(fromChild);
            ::execl(executable.c_str(), executable.c_str(), static_cast<char*>(nullptr));
            std::cerr << "imb-guest-probe: exec failed: " << std::strerror(errno) << '\n';
            _exit(127);
        }

        ::close(toChild[0]);
        ::close(fromChild[1]);
        writeFD_ = toChild[1];
        readFD_ = fromChild[0];
    }

    ~ChildProcessTransport() override {
        if (writeFD_ >= 0) ::close(writeFD_);
        if (readFD_ >= 0) ::close(readFD_);
        if (child_ > 0) {
            int status = 0;
            while (::waitpid(child_, &status, 0) < 0 && errno == EINTR) {}
        }
    }

    void writeAll(const std::uint8_t* data, std::size_t size) override {
        writeToFD(writeFD_, data, size);
    }

    void readAll(std::uint8_t* data, std::size_t size) override {
        readFromFD(readFD_, data, size);
    }

private:
    static void closePipe(int fds[2]) {
        ::close(fds[0]);
        ::close(fds[1]);
    }

    static void writeToFD(int fd, const std::uint8_t* data, std::size_t size) {
        std::size_t offset = 0;
        while (offset < size) {
            const ssize_t written = ::write(fd, data + offset, size - offset);
            if (written < 0 && errno == EINTR) continue;
            if (written <= 0) throw std::runtime_error("host pipe write failed: " + std::string(std::strerror(errno)));
            offset += static_cast<std::size_t>(written);
        }
    }

    static void readFromFD(int fd, std::uint8_t* data, std::size_t size) {
        std::size_t offset = 0;
        while (offset < size) {
            const ssize_t count = ::read(fd, data + offset, size - offset);
            if (count < 0 && errno == EINTR) continue;
            if (count < 0) throw std::runtime_error("host pipe read failed: " + std::string(std::strerror(errno)));
            if (count == 0) throw std::runtime_error("host process closed the protocol stream");
            offset += static_cast<std::size_t>(count);
        }
    }

    pid_t child_ = -1;
    int readFD_ = -1;
    int writeFD_ = -1;
};

struct Frame {
    imb_message_header header{};
    imb::Bytes payload;
};

class Client {
public:
    Client(Transport& transport, std::ostream& output, std::optional<std::string> imageOutput)
        : transport_(transport), output_(output), imageOutput_(std::move(imageOutput)) {}

    void run() {
        hello();
        queryCapabilities();
        ping();
        const std::uint64_t required = IMB_CAP_METAL_BUFFER
            | IMB_CAP_METAL_COMPUTE
            | IMB_CAP_RESOURCE_IO
            | IMB_CAP_REAL_FENCE;
        if ((capabilityBits_ & required) == required) {
            runMetalCompute();
        } else {
            output_ << "METAL_COMPUTE skipped: host did not advertise the verified data path\n";
        }
        const std::uint64_t rasterRequired = IMB_CAP_METAL_IMAGE
            | IMB_CAP_METAL_RASTER
            | IMB_CAP_RESOURCE_IO
            | IMB_CAP_REAL_FENCE;
        if ((capabilityBits_ & rasterRequired) == rasterRequired) {
            runMetalRaster();
        } else {
            output_ << "METAL_RASTER skipped: host did not advertise the verified image path\n";
        }
        shutdown();
    }

private:
    Frame exchange(std::uint16_t type, imb::Bytes payload = {}, std::uint64_t resourceID = 0) {
        const std::uint64_t requestID = nextRequestID_++;
        imb_message_header header{
            IMB_PROTOCOL_MAGIC,
            IMB_PROTOCOL_VERSION_MAJOR,
            IMB_PROTOCOL_VERSION_MINOR,
            type,
            IMB_FLAG_NONE,
            static_cast<std::uint32_t>(payload.size()),
            requestID,
            resourceID,
        };
        const auto headerBytes = imb::encodeHeader(header);
        transport_.writeAll(headerBytes.data(), headerBytes.size());
        if (!payload.empty()) transport_.writeAll(payload.data(), payload.size());

        imb::Bytes replyHeaderBytes(IMB_PROTOCOL_HEADER_SIZE);
        transport_.readAll(replyHeaderBytes.data(), replyHeaderBytes.size());
        Frame reply;
        reply.header = imb::decodeHeader(replyHeaderBytes);
        if (reply.header.magic != IMB_PROTOCOL_MAGIC) throw std::runtime_error("invalid reply magic");
        if ((reply.header.flags & IMB_FLAG_RESPONSE) == 0) throw std::runtime_error("reply flag is missing");
        if (reply.header.request_id != requestID) throw std::runtime_error("reply request ID mismatch");
        if (reply.header.payload_length > IMB_PROTOCOL_MAX_PAYLOAD) throw std::runtime_error("reply payload is too large");
        reply.payload.resize(reply.header.payload_length);
        if (!reply.payload.empty()) transport_.readAll(reply.payload.data(), reply.payload.size());
        if (reply.header.message_type == IMB_MSG_ERROR) throwProtocolError(reply.payload);
        return reply;
    }

    static void expectType(const Frame& frame, std::uint16_t expected) {
        if (frame.header.message_type != expected) {
            throw std::runtime_error("unexpected reply type " + std::to_string(frame.header.message_type));
        }
    }

    [[noreturn]] static void throwProtocolError(const imb::Bytes& payload) {
        if (payload.size() < sizeof(imb_error_payload)) throw std::runtime_error("malformed protocol error");
        const auto code = imb::readLittleEndian<std::uint32_t>(payload, 0);
        const auto length = imb::readLittleEndian<std::uint32_t>(payload, 4);
        if (length > payload.size() - sizeof(imb_error_payload)) throw std::runtime_error("malformed protocol error string");
        const std::string message(payload.begin() + sizeof(imb_error_payload), payload.begin() + sizeof(imb_error_payload) + length);
        throw std::runtime_error("host protocol error " + std::to_string(code) + ": " + message);
    }

    void hello() {
        imb::Bytes payload;
        imb::appendLittleEndian(payload, std::uint16_t{IMB_PROTOCOL_VERSION_MAJOR});
        imb::appendLittleEndian(payload, std::uint16_t{IMB_PROTOCOL_VERSION_MINOR});
        imb::appendLittleEndian(payload, std::uint16_t{IMB_PROTOCOL_VERSION_MAJOR});
        imb::appendLittleEndian(payload, std::uint16_t{IMB_PROTOCOL_VERSION_MINOR});
        const auto reply = exchange(IMB_MSG_HELLO, std::move(payload));
        expectType(reply, IMB_MSG_HELLO_REPLY);
        if (reply.payload.size() != sizeof(imb_hello_reply_payload)) throw std::runtime_error("invalid HELLO_REPLY size");
        const auto major = imb::readLittleEndian<std::uint16_t>(reply.payload, 0);
        const auto minor = imb::readLittleEndian<std::uint16_t>(reply.payload, 2);
        if (major != IMB_PROTOCOL_VERSION_MAJOR || minor != IMB_PROTOCOL_VERSION_MINOR) {
            throw std::runtime_error("host selected an unsupported version");
        }
        output_ << "HELLO negotiated IMB " << major << '.' << minor << '\n';
    }

    void queryCapabilities() {
        const auto reply = exchange(IMB_MSG_QUERY_CAPABILITIES);
        expectType(reply, IMB_MSG_CAPABILITIES_REPLY);
        if (reply.payload.size() < sizeof(imb_capabilities_payload)) throw std::runtime_error("invalid capabilities payload");
        const auto bits = imb::readLittleEndian<std::uint64_t>(reply.payload, 0);
        capabilityBits_ = bits;
        const auto maxBuffer = imb::readLittleEndian<std::uint64_t>(reply.payload, 8);
        const auto nameLength = imb::readLittleEndian<std::uint32_t>(reply.payload, 16);
        if (nameLength > reply.payload.size() - sizeof(imb_capabilities_payload)) throw std::runtime_error("invalid GPU name length");
        const std::string name(
            reply.payload.begin() + sizeof(imb_capabilities_payload),
            reply.payload.begin() + sizeof(imb_capabilities_payload) + nameLength
        );
        output_ << "CAPABILITIES bits=0x" << std::hex << bits << std::dec
                << " metal_device=\"" << (name.empty() ? "none" : name)
                << "\" max_buffer_length=" << maxBuffer << '\n';
    }

    void ping() {
        const std::string value = "imb-guest-probe";
        imb::Bytes payload(value.begin(), value.end());
        const auto reply = exchange(IMB_MSG_PING, payload);
        expectType(reply, IMB_MSG_PONG);
        if (reply.payload != payload) throw std::runtime_error("PONG did not echo PING");
        output_ << "PING/PONG ok\n";
    }

    void runMetalCompute() {
        const auto resource = createResource(16);
        writeResource(resource, {1, 2, 3, 4});
        const auto fence = submitAdd(resource, 4, 5);
        waitFence(fence);
        const auto values = readResourceBytes(resource, 16);
        const imb::Bytes expected{6, 0, 0, 0, 7, 0, 0, 0, 8, 0, 0, 0, 9, 0, 0, 0};
        if (values != expected) {
            throw std::runtime_error("Metal ADD_U32 readback mismatch");
        }
        output_ << "METAL_COMPUTE ADD_U32 verified: [1,2,3,4] + 5 = [6,7,8,9]\n";
        destroyResource(resource);
    }

    void runMetalRaster() {
        constexpr std::uint32_t width = 64;
        constexpr std::uint32_t height = 64;
        constexpr std::uint32_t clearRGBA8 = 0xff201810;
        const auto image = createImage(width, height);
        const auto fence = submitTriangle(image, clearRGBA8);
        waitFence(fence);
        const auto pixels = readResourceBytes(image, static_cast<std::uint64_t>(width) * height * 4);

        const std::uint8_t clear[4]{0x10, 0x18, 0x20, 0xff};
        if (!std::equal(std::begin(clear), std::end(clear), pixels.begin())) {
            throw std::runtime_error("Metal triangle clear-color corner mismatch");
        }
        std::size_t changedPixels = 0;
        for (std::size_t offset = 0; offset < pixels.size(); offset += 4) {
            if (pixels[offset + 3] != 0xff) {
                throw std::runtime_error("Metal triangle image contains a non-opaque pixel");
            }
            if (!std::equal(std::begin(clear), std::end(clear), pixels.begin() + static_cast<std::ptrdiff_t>(offset))) {
                ++changedPixels;
            }
        }
        if (changedPixels < 512 || changedPixels >= width * height) {
            throw std::runtime_error("Metal triangle changed-pixel coverage is invalid: " + std::to_string(changedPixels));
        }
        if (imageOutput_) {
            writePPM(*imageOutput_, pixels, width, height);
        }
        output_ << "METAL_RASTER DRAW_TRIANGLE verified: " << width << 'x' << height
                << " RGBA8 changed_pixels=" << changedPixels;
        if (imageOutput_) output_ << " image=\"" << *imageOutput_ << '"';
        output_ << '\n';
        destroyResource(image);
    }

    std::uint64_t createResource(std::uint64_t size) {
        imb::Bytes payload;
        imb::appendLittleEndian(payload, size);
        imb::appendLittleEndian(payload, std::uint32_t{IMB_RESOURCE_BUFFER});
        imb::appendLittleEndian(payload, std::uint32_t{0});
        const auto reply = exchange(IMB_MSG_CREATE_RESOURCE, std::move(payload));
        expectType(reply, IMB_MSG_CREATE_RESOURCE);
        if (reply.header.resource_id == 0) throw std::runtime_error("host returned resource ID zero");
        output_ << "CREATE_RESOURCE id=" << reply.header.resource_id << " (Metal buffer, " << size << " bytes)\n";
        return reply.header.resource_id;
    }

    std::uint64_t createImage(std::uint32_t width, std::uint32_t height) {
        const std::uint64_t size = static_cast<std::uint64_t>(width) * height * 4;
        imb::Bytes payload;
        imb::appendLittleEndian(payload, size);
        imb::appendLittleEndian(payload, std::uint32_t{IMB_RESOURCE_IMAGE});
        imb::appendLittleEndian(payload, std::uint32_t{0});
        imb::appendLittleEndian(payload, width);
        imb::appendLittleEndian(payload, height);
        imb::appendLittleEndian(payload, std::uint32_t{IMB_IMAGE_FORMAT_RGBA8_UNORM});
        imb::appendLittleEndian(payload, std::uint32_t{0});
        const auto reply = exchange(IMB_MSG_CREATE_RESOURCE, std::move(payload));
        expectType(reply, IMB_MSG_CREATE_RESOURCE);
        if (reply.header.resource_id == 0) throw std::runtime_error("host returned image resource ID zero");
        output_ << "CREATE_RESOURCE id=" << reply.header.resource_id << " (Metal RGBA8 image, "
                << width << 'x' << height << ")\n";
        return reply.header.resource_id;
    }

    void writeResource(std::uint64_t resource, const std::vector<std::uint32_t>& values) {
        imb::Bytes payload;
        imb::appendLittleEndian(payload, std::uint64_t{0});
        imb::appendLittleEndian(payload, static_cast<std::uint32_t>(values.size() * sizeof(std::uint32_t)));
        imb::appendLittleEndian(payload, std::uint32_t{0});
        for (const auto value : values) imb::appendLittleEndian(payload, value);
        const auto reply = exchange(IMB_MSG_WRITE_RESOURCE, std::move(payload), resource);
        expectType(reply, IMB_MSG_WRITE_RESOURCE);
        output_ << "WRITE_RESOURCE id=" << resource << " bytes=" << values.size() * sizeof(std::uint32_t) << '\n';
    }

    std::uint64_t submitAdd(
        std::uint64_t resource,
        std::uint32_t elementCount,
        std::uint32_t addend
    ) {
        imb::Bytes payload;
        imb::appendLittleEndian(payload, std::uint16_t{IMB_COMMAND_ADD_U32});
        imb::appendLittleEndian(payload, std::uint16_t{0});
        imb::appendLittleEndian(payload, std::uint32_t{0});
        imb::appendLittleEndian(payload, elementCount);
        imb::appendLittleEndian(payload, addend);
        const auto reply = exchange(IMB_MSG_SUBMIT_COMMAND, std::move(payload), resource);
        expectType(reply, IMB_MSG_SUBMIT_COMMAND);
        if (reply.header.resource_id == 0) throw std::runtime_error("host returned fence ID zero");
        output_ << "SUBMIT_COMMAND ADD_U32 fence=" << reply.header.resource_id << '\n';
        return reply.header.resource_id;
    }

    std::uint64_t submitTriangle(std::uint64_t image, std::uint32_t clearRGBA8) {
        imb::Bytes payload;
        imb::appendLittleEndian(payload, std::uint16_t{IMB_COMMAND_DRAW_TRIANGLE});
        imb::appendLittleEndian(payload, std::uint16_t{0});
        imb::appendLittleEndian(payload, std::uint32_t{0});
        imb::appendLittleEndian(payload, clearRGBA8);
        imb::appendLittleEndian(payload, std::uint32_t{0});
        const auto reply = exchange(IMB_MSG_SUBMIT_COMMAND, std::move(payload), image);
        expectType(reply, IMB_MSG_SUBMIT_COMMAND);
        if (reply.header.resource_id == 0) throw std::runtime_error("host returned raster fence ID zero");
        output_ << "SUBMIT_COMMAND DRAW_TRIANGLE fence=" << reply.header.resource_id << '\n';
        return reply.header.resource_id;
    }

    void waitFence(std::uint64_t fence) {
        const auto reply = exchange(IMB_MSG_WAIT_FENCE, {}, fence);
        expectType(reply, IMB_MSG_WAIT_FENCE);
        if (reply.payload.size() != sizeof(imb_wait_fence_reply_payload)
            || imb::readLittleEndian<std::uint32_t>(reply.payload, 0) != 1) {
            throw std::runtime_error("Metal fence was not signaled");
        }
        output_ << "WAIT_FENCE signaled (real Metal command buffer)\n";
    }

    imb::Bytes readResourceBytes(std::uint64_t resource, std::uint64_t byteCount) {
        imb::Bytes payload;
        imb::appendLittleEndian(payload, std::uint64_t{0});
        imb::appendLittleEndian(payload, byteCount);
        const auto reply = exchange(IMB_MSG_READ_RESOURCE, std::move(payload), resource);
        expectType(reply, IMB_MSG_READ_RESOURCE);
        if (reply.payload.size() != byteCount) throw std::runtime_error("READ_RESOURCE returned the wrong byte count");
        output_ << "READ_RESOURCE id=" << resource << " bytes=" << byteCount << '\n';
        return reply.payload;
    }

    static void writePPM(
        const std::string& path,
        const imb::Bytes& rgba,
        std::uint32_t width,
        std::uint32_t height
    ) {
        std::ofstream output(path, std::ios::binary | std::ios::trunc);
        if (!output) throw std::runtime_error("could not create raster output " + path);
        output << "P6\n" << width << ' ' << height << "\n255\n";
        for (std::size_t offset = 0; offset < rgba.size(); offset += 4) {
            output.put(static_cast<char>(rgba[offset]));
            output.put(static_cast<char>(rgba[offset + 1]));
            output.put(static_cast<char>(rgba[offset + 2]));
        }
        if (!output) throw std::runtime_error("could not finish raster output " + path);
    }

    void destroyResource(std::uint64_t resource) {
        const auto reply = exchange(IMB_MSG_DESTROY_RESOURCE, {}, resource);
        expectType(reply, IMB_MSG_DESTROY_RESOURCE);
        output_ << "DESTROY_RESOURCE id=" << resource << '\n';
    }

    void shutdown() {
        const auto reply = exchange(IMB_MSG_SHUTDOWN);
        expectType(reply, IMB_MSG_SHUTDOWN);
        output_ << "SHUTDOWN ok\n";
    }

    Transport& transport_;
    std::ostream& output_;
    std::uint64_t capabilityBits_ = 0;
    std::uint64_t nextRequestID_ = 1;
    std::optional<std::string> imageOutput_;
};

void usage(const char* program) {
    std::cerr << "usage: " << program
              << " (--host /path/to/imb-host | --stdio | --vsock-listen <port>)"
              << " [--image-output /path/to/triangle.ppm]\n";
}

}  // namespace

int main(int argc, char** argv) {
    std::signal(SIGPIPE, SIG_IGN);
    try {
        std::unique_ptr<Transport> transport;
        std::ostream* statusOutput = &std::cout;
        std::optional<std::string> mode;
        std::optional<std::string> modeValue;
        std::optional<std::string> imageOutput;
        for (int index = 1; index < argc; ++index) {
            const std::string_view argument(argv[index]);
            if (argument == "--stdio" && !mode) {
                mode = "stdio";
            } else if ((argument == "--host" || argument == "--vsock-listen") && !mode && index + 1 < argc) {
                mode = argument == "--host" ? "host" : "vsock";
                modeValue = argv[++index];
            } else if (argument == "--image-output" && !imageOutput && index + 1 < argc) {
                imageOutput = argv[++index];
            } else {
                usage(argv[0]);
                return 2;
            }
        }

        if (mode == "host" && modeValue) {
            transport = std::make_unique<ChildProcessTransport>(*modeValue);
        } else if (mode == "stdio") {
            transport = std::make_unique<FileDescriptorTransport>(STDIN_FILENO, STDOUT_FILENO, false);
            statusOutput = &std::cerr;
        } else if (mode == "vsock" && modeValue) {
#if defined(__linux__)
            std::size_t parsedCharacters = 0;
            const unsigned long parsedPort = std::stoul(*modeValue, &parsedCharacters, 10);
            if (parsedCharacters != modeValue->size() || parsedPort == 0 || parsedPort > UINT32_MAX) {
                throw std::runtime_error("vsock port must be between 1 and 4294967295");
            }
            transport = std::make_unique<VsockListenerTransport>(static_cast<std::uint32_t>(parsedPort));
#else
            throw std::runtime_error("--vsock-listen is available only in a Linux guest build");
#endif
        } else {
            usage(argv[0]);
            return 2;
        }
        Client client(*transport, *statusOutput, imageOutput);
        client.run();
        *statusOutput << "imb-guest-probe: protocol and Metal data-path checks completed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "imb-guest-probe: " << error.what() << '\n';
        return 1;
    }
}
