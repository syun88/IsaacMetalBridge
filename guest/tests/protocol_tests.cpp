#include "imb_protocol.h"
#include "imb_wire.hpp"

#include <cstdint>
#include <iostream>
#include <stdexcept>

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

}  // namespace

int main() {
    static_assert(sizeof(imb_message_header) == 32);
    static_assert(sizeof(imb_hello_payload) == 8);
    static_assert(sizeof(imb_capabilities_payload) == 24);
    static_assert(sizeof(imb_create_image_resource_payload) == 32);
    static_assert(sizeof(imb_create_compute_pipeline_payload) == 16);
    static_assert(sizeof(imb_create_acceleration_structure_payload) == 16);
    static_assert(sizeof(imb_build_primitive_acceleration_structure_payload) == 16);
    static_assert(sizeof(imb_primitive_acceleration_structure_geometry_payload) == 72);
    static_assert(sizeof(imb_build_instance_acceleration_structure_payload) == 16);
    static_assert(sizeof(imb_acceleration_structure_instance_payload) == 80);
    static_assert(sizeof(imb_trace_rays_command_payload) == 168);
    static_assert(sizeof(imb_add_u32_command_payload) == 16);
    static_assert(sizeof(imb_draw_triangle_command_payload) == 16);
    static_assert(sizeof(imb_draw_indexed_ui_command_payload) == 56);
    static_assert(sizeof(imb_ui_draw_payload) == 40);
    static_assert(sizeof(imb_write_resource_payload) == 16);
    static_assert(sizeof(imb_read_resource_payload) == 16);

    require(IMB_PROTOCOL_VERSION_MAJOR == 1 && IMB_PROTOCOL_VERSION_MINOR == 13, "unexpected protocol version");
    require((IMB_CAP_METAL_COMPUTE & IMB_CAP_RESOURCE_IO) == 0, "capability bits overlap");
    require((IMB_CAP_METAL_IMAGE & IMB_CAP_METAL_RASTER) == 0, "raster capability bits overlap");
    require((IMB_CAP_METAL_RASTER & IMB_CAP_METAL_UI_RASTER) == 0, "UI raster capability bits overlap");
    require((IMB_CAP_METAL_UI_RASTER & IMB_CAP_METAL_SPIRV_COMPUTE) == 0, "SPIR-V capability bits overlap");
    require((IMB_CAP_METAL_SPIRV_COMPUTE & IMB_CAP_METAL_ACCELERATION_STRUCTURE) == 0, "acceleration structure capability bits overlap");
    require((IMB_CAP_METAL_ACCELERATION_STRUCTURE & IMB_CAP_METAL_RAY_DISPATCH) == 0, "ray dispatch capability bits overlap");

    const imb_message_header header{
        IMB_PROTOCOL_MAGIC,
        IMB_PROTOCOL_VERSION_MAJOR,
        IMB_PROTOCOL_VERSION_MINOR,
        IMB_MSG_PING,
        IMB_FLAG_NONE,
        4,
        UINT64_C(0x0102030405060708),
        0,
    };
    const auto encoded = imb::encodeHeader(header);
    require(encoded.size() == IMB_PROTOCOL_HEADER_SIZE, "encoded header size mismatch");
    require(encoded[0] == 0x49 && encoded[1] == 0x4d && encoded[2] == 0x42 && encoded[3] == 0x31, "magic byte order mismatch");
    const auto decoded = imb::decodeHeader(encoded);
    require(decoded.magic == header.magic, "decoded magic mismatch");
    require(decoded.message_type == IMB_MSG_PING, "decoded type mismatch");
    require(decoded.request_id == header.request_id, "decoded request ID mismatch");
    require(decoded.payload_length == 4, "decoded payload length mismatch");

    require(imb::versionRangeIncludes(1, 0, 1, 0, 1, 0), "exact version rejected");
    require(imb::versionRangeIncludes(0, 9, 2, 0, 1, 0), "inclusive version range rejected");
    require(!imb::versionRangeIncludes(2, 0, 3, 0, 1, 0), "incompatible version accepted");

    bool rejected = false;
    try {
        (void)imb::decodeHeader(imb::Bytes(31));
    } catch (...) {
        rejected = true;
    }
    require(rejected, "short header was not rejected");

    std::cout << "IMB C/C++ protocol ABI tests passed\n";
    return 0;
}
