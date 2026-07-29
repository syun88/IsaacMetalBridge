#ifndef IMB_WIRE_HPP
#define IMB_WIRE_HPP

#include "imb_protocol.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace imb {

using Bytes = std::vector<std::uint8_t>;

template <typename T>
void appendLittleEndian(Bytes& output, T value) {
    static_assert(std::is_integral_v<T> && std::is_unsigned_v<T>);
    for (std::size_t i = 0; i < sizeof(T); ++i) {
        output.push_back(static_cast<std::uint8_t>((value >> (i * 8)) & 0xffu));
    }
}

template <typename T>
T readLittleEndian(const Bytes& input, std::size_t offset) {
    static_assert(std::is_integral_v<T> && std::is_unsigned_v<T>);
    if (offset > input.size() || sizeof(T) > input.size() - offset) {
        throw std::runtime_error("integer exceeds payload boundary");
    }
    T result = 0;
    for (std::size_t i = 0; i < sizeof(T); ++i) {
        result |= static_cast<T>(input[offset + i]) << (i * 8);
    }
    return result;
}

inline Bytes encodeHeader(const imb_message_header& header) {
    Bytes output;
    output.reserve(IMB_PROTOCOL_HEADER_SIZE);
    appendLittleEndian(output, header.magic);
    appendLittleEndian(output, header.version_major);
    appendLittleEndian(output, header.version_minor);
    appendLittleEndian(output, header.message_type);
    appendLittleEndian(output, header.flags);
    appendLittleEndian(output, header.payload_length);
    appendLittleEndian(output, header.request_id);
    appendLittleEndian(output, header.resource_id);
    return output;
}

inline imb_message_header decodeHeader(const Bytes& input) {
    if (input.size() != IMB_PROTOCOL_HEADER_SIZE) {
        throw std::runtime_error("IMB header must be 32 bytes");
    }
    return imb_message_header{
        readLittleEndian<std::uint32_t>(input, 0),
        readLittleEndian<std::uint16_t>(input, 4),
        readLittleEndian<std::uint16_t>(input, 6),
        readLittleEndian<std::uint16_t>(input, 8),
        readLittleEndian<std::uint16_t>(input, 10),
        readLittleEndian<std::uint32_t>(input, 12),
        readLittleEndian<std::uint64_t>(input, 16),
        readLittleEndian<std::uint64_t>(input, 24),
    };
}

inline bool versionRangeIncludes(
    std::uint16_t minMajor,
    std::uint16_t minMinor,
    std::uint16_t maxMajor,
    std::uint16_t maxMinor,
    std::uint16_t major,
    std::uint16_t minor
) {
    const bool aboveMinimum = minMajor < major || (minMajor == major && minMinor <= minor);
    const bool belowMaximum = maxMajor > major || (maxMajor == major && maxMinor >= minor);
    return aboveMinimum && belowMaximum;
}

}  // namespace imb

#endif
