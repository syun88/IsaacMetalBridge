#ifndef IMB_PROTOCOL_H
#define IMB_PROTOCOL_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define IMB_PROTOCOL_MAGIC UINT32_C(0x31424d49) /* little-endian bytes: I M B 1 */
#define IMB_PROTOCOL_VERSION_MAJOR UINT16_C(1)
#define IMB_PROTOCOL_VERSION_MINOR UINT16_C(16)
#define IMB_PROTOCOL_HEADER_SIZE UINT32_C(32)
#define IMB_PROTOCOL_MAX_PAYLOAD UINT32_C(16777216)

enum imb_message_type {
    IMB_MSG_HELLO = 1,
    IMB_MSG_HELLO_REPLY = 2,
    IMB_MSG_QUERY_CAPABILITIES = 3,
    IMB_MSG_CAPABILITIES_REPLY = 4,
    IMB_MSG_PING = 5,
    IMB_MSG_PONG = 6,
    IMB_MSG_CREATE_RESOURCE = 7,
    IMB_MSG_DESTROY_RESOURCE = 8,
    IMB_MSG_SUBMIT_COMMAND = 9,
    IMB_MSG_WAIT_FENCE = 10,
    IMB_MSG_ERROR = 11,
    IMB_MSG_SHUTDOWN = 12,
    IMB_MSG_WRITE_RESOURCE = 13,
    IMB_MSG_READ_RESOURCE = 14,
    IMB_MSG_CREATE_COMPUTE_PIPELINE = 15,
    IMB_MSG_CREATE_ACCELERATION_STRUCTURE = 16,
    IMB_MSG_BUILD_PRIMITIVE_ACCELERATION_STRUCTURE = 17,
    IMB_MSG_BUILD_INSTANCE_ACCELERATION_STRUCTURE = 18,
    IMB_MSG_QUERY_SPARSE_IMAGE_PROPERTIES = 19,
    IMB_MSG_UPDATE_SPARSE_IMAGE_MAPPING = 20
};

enum imb_message_flags {
    IMB_FLAG_NONE = 0,
    IMB_FLAG_RESPONSE = 1u << 0
};

enum imb_error_code {
    IMB_ERROR_NONE = 0,
    IMB_ERROR_INVALID_MAGIC = 1,
    IMB_ERROR_INVALID_HEADER = 2,
    IMB_ERROR_UNSUPPORTED_VERSION = 3,
    IMB_ERROR_UNSUPPORTED_MESSAGE = 4,
    IMB_ERROR_INVALID_PAYLOAD = 5,
    IMB_ERROR_HANDSHAKE_REQUIRED = 6,
    IMB_ERROR_RESOURCE_NOT_FOUND = 7,
    IMB_ERROR_RESOURCE_IN_USE = 8,
    IMB_ERROR_UNSUPPORTED_COMMAND = 9,
    IMB_ERROR_INTERNAL = 10,
    IMB_ERROR_BACKEND_UNAVAILABLE = 11,
    IMB_ERROR_OUT_OF_BOUNDS = 12,
    IMB_ERROR_GPU_FAILURE = 13
};

enum imb_capability_bits {
    IMB_CAP_METAL_AVAILABLE = UINT64_C(1) << 0,
    IMB_CAP_UNIFIED_MEMORY = UINT64_C(1) << 1,
    IMB_CAP_RAY_TRACING = UINT64_C(1) << 2,
    IMB_CAP_NOOP_COMMAND = UINT64_C(1) << 3,
    IMB_CAP_SIMULATED_FENCE = UINT64_C(1) << 4,
    IMB_CAP_METAL_BUFFER = UINT64_C(1) << 5,
    IMB_CAP_METAL_COMPUTE = UINT64_C(1) << 6,
    IMB_CAP_RESOURCE_IO = UINT64_C(1) << 7,
    IMB_CAP_REAL_FENCE = UINT64_C(1) << 8,
    IMB_CAP_METAL_IMAGE = UINT64_C(1) << 9,
    IMB_CAP_METAL_RASTER = UINT64_C(1) << 10,
    IMB_CAP_METAL_UI_RASTER = UINT64_C(1) << 11,
    IMB_CAP_METAL_SPIRV_COMPUTE = UINT64_C(1) << 12,
    IMB_CAP_METAL_ACCELERATION_STRUCTURE = UINT64_C(1) << 13,
    IMB_CAP_METAL_RAY_DISPATCH = UINT64_C(1) << 14,
    IMB_CAP_METAL_SPARSE_IMAGE = UINT64_C(1) << 15
};

enum imb_resource_kind {
    IMB_RESOURCE_BUFFER = 1,
    IMB_RESOURCE_IMAGE = 2,
    IMB_RESOURCE_COMPUTE_PIPELINE = 3,
    IMB_RESOURCE_ACCELERATION_STRUCTURE = 4
};

enum imb_image_format {
    IMB_IMAGE_FORMAT_RGBA8_UNORM = 1,
    IMB_IMAGE_FORMAT_BGRA8_UNORM = 2,
    IMB_IMAGE_FORMAT_BC3_UNORM = 3,
    IMB_IMAGE_FORMAT_R16_UNORM = 4,
    IMB_IMAGE_FORMAT_RGBA16_UNORM = 5,
    IMB_IMAGE_FORMAT_BC3_SRGB = 6,
    IMB_IMAGE_FORMAT_BC5_UNORM = 7,
    IMB_IMAGE_FORMAT_RGBA8_UINT = 8,
    IMB_IMAGE_FORMAT_RGBA8_SNORM = 9,
    IMB_IMAGE_FORMAT_RGBA8_SRGB = 10
};

enum imb_image_options {
    IMB_IMAGE_OPTION_NONE = 0,
    IMB_IMAGE_OPTION_SPARSE = 1u << 0,
    /*
     * Ordinary 3D images retain the 32-byte CREATE_RESOURCE payload.  Bit 1
     * selects a 3D Metal texture and the upper 16 bits carry VkExtent3D.depth.
     * Sparse resources keep their existing extended payload and may not set
     * these fields.
     */
    IMB_IMAGE_OPTION_3D = 1u << 1,
    IMB_IMAGE_OPTION_DEPTH_SHIFT = 16,
    IMB_IMAGE_OPTION_DEPTH_MASK = 0xffffu << IMB_IMAGE_OPTION_DEPTH_SHIFT
};

enum imb_texture_type {
    IMB_TEXTURE_TYPE_2D = 1
};

enum imb_sparse_mapping_mode {
    IMB_SPARSE_MAPPING_MAP = 0,
    IMB_SPARSE_MAPPING_UNMAP = 1
};

enum imb_command_kind {
    IMB_COMMAND_NOOP = 0,
    IMB_COMMAND_ADD_U32 = 1,
    IMB_COMMAND_DRAW_TRIANGLE = 2,
    IMB_COMMAND_DRAW_INDEXED_UI = 3,
    IMB_COMMAND_TRACE_RAYS = 4,
    IMB_COMMAND_DISPATCH_COMPUTE = 5
};

enum imb_compute_binding_kind {
    IMB_COMPUTE_BINDING_BUFFER_READ = 1,
    IMB_COMPUTE_BINDING_BUFFER_READ_WRITE = 2,
    IMB_COMPUTE_BINDING_TEXTURE_READ = 3,
    IMB_COMPUTE_BINDING_TEXTURE_READ_WRITE = 4,
    IMB_COMPUTE_BINDING_TEXEL_BUFFER_READ = 5,
    IMB_COMPUTE_BINDING_TEXEL_BUFFER_READ_WRITE = 6,
    /* Inline VkSamplerCreateInfo state; this binding has no resource ID. */
    IMB_COMPUTE_BINDING_SAMPLER = 7
};

/*
 * IMB_COMPUTE_BINDING_SAMPLER keeps the fixed 48-byte binding record:
 *
 * format bits:
 *   0 magFilter, 1 minFilter, 2 mipmapMode,
 *   3..5 addressModeU, 6..8 addressModeV, 9..11 addressModeW,
 *   12 anisotropyEnable, 13 compareEnable, 14..16 compareOp,
 *   17 unnormalizedCoordinates, 18..20 borderColor.
 * resource_id: zero.
 * offset: low/high float32 bits are minLod/maxLod.
 * length: low/high float32 bits are mipLodBias/maxAnisotropy.
 */
#define IMB_COMPUTE_SAMPLER_MAG_FILTER_SHIFT UINT32_C(0)
#define IMB_COMPUTE_SAMPLER_MIN_FILTER_SHIFT UINT32_C(1)
#define IMB_COMPUTE_SAMPLER_MIPMAP_MODE_SHIFT UINT32_C(2)
#define IMB_COMPUTE_SAMPLER_ADDRESS_U_SHIFT UINT32_C(3)
#define IMB_COMPUTE_SAMPLER_ADDRESS_V_SHIFT UINT32_C(6)
#define IMB_COMPUTE_SAMPLER_ADDRESS_W_SHIFT UINT32_C(9)
#define IMB_COMPUTE_SAMPLER_ANISOTROPY_ENABLE_SHIFT UINT32_C(12)
#define IMB_COMPUTE_SAMPLER_COMPARE_ENABLE_SHIFT UINT32_C(13)
#define IMB_COMPUTE_SAMPLER_COMPARE_OP_SHIFT UINT32_C(14)
#define IMB_COMPUTE_SAMPLER_UNNORMALIZED_SHIFT UINT32_C(17)
#define IMB_COMPUTE_SAMPLER_BORDER_COLOR_SHIFT UINT32_C(18)
#define IMB_COMPUTE_SAMPLER_OPTIONS_MASK UINT32_C(0x001fffff)

enum imb_compute_pipeline_flags {
    IMB_COMPUTE_PIPELINE_FLAG_NONE = 0,
    /* The translated entry point evaluates values through software binary64. */
    IMB_COMPUTE_PIPELINE_FLAG_SOFTWARE_FP64_EXECUTION_REQUIRED = 1u << 0,
    /* Logical invocations are serialized to preserve 64-bit compare-exchange. */
    IMB_COMPUTE_PIPELINE_FLAG_SERIALIZED_ATOMIC64_EXECUTION_REQUIRED = 1u << 1
};

enum imb_trace_rays_options {
    IMB_TRACE_RAYS_OPTION_NONE = 0,
    IMB_TRACE_RAYS_OPTION_LIVE_CAMERA = 1u << 0,
    IMB_TRACE_RAYS_OPTION_LIVE_SPHERE_LIGHT = 1u << 1,
    IMB_TRACE_RAYS_OPTION_LIVE_DISTANT_LIGHT = 1u << 2,
    IMB_TRACE_RAYS_OPTION_LIVE_DOME_LIGHT = 1u << 3,
    /* Draw the renderer-owned empty-stage guide grid; AS resource ID is zero. */
    IMB_TRACE_RAYS_OPTION_EMPTY_STAGE_GRID = 1u << 4
};

enum imb_acceleration_structure_vertex_format {
    IMB_ACCELERATION_STRUCTURE_VERTEX_FORMAT_FLOAT3_POSITION = 1,
    /* float3 position followed by float3 authored corner normal */
    IMB_ACCELERATION_STRUCTURE_VERTEX_FORMAT_FLOAT3_POSITION_NORMAL = 2,
    /* format 2 followed by float2 material UV */
    IMB_ACCELERATION_STRUCTURE_VERTEX_FORMAT_FLOAT3_POSITION_NORMAL_UV = 3,
    /* format 3 followed by float roughness and float metallic */
    IMB_ACCELERATION_STRUCTURE_VERTEX_FORMAT_FLOAT3_POSITION_NORMAL_UV_MATERIAL = 4,
    /* format 4 followed by float3 emission color and float intensity */
    IMB_ACCELERATION_STRUCTURE_VERTEX_FORMAT_FLOAT3_POSITION_NORMAL_UV_MATERIAL_EMISSION = 5,
    /* format 5 followed by float3 tangent and float3 bitangent */
    IMB_ACCELERATION_STRUCTURE_VERTEX_FORMAT_FLOAT3_POSITION_NORMAL_UV_MATERIAL_EMISSION_TBN = 6
};

#pragma pack(push, 1)

typedef struct imb_message_header {
    uint32_t magic;
    uint16_t version_major;
    uint16_t version_minor;
    uint16_t message_type;
    uint16_t flags;
    uint32_t payload_length;
    uint64_t request_id;
    uint64_t resource_id;
} imb_message_header;

typedef struct imb_hello_payload {
    uint16_t min_major;
    uint16_t min_minor;
    uint16_t max_major;
    uint16_t max_minor;
} imb_hello_payload;

typedef struct imb_hello_reply_payload {
    uint16_t selected_major;
    uint16_t selected_minor;
    uint32_t status;
} imb_hello_reply_payload;

typedef struct imb_capabilities_payload {
    uint64_t capability_bits;
    uint64_t max_buffer_length;
    uint32_t gpu_name_length;
    uint32_t reserved;
    /* gpu_name_length bytes of UTF-8 follow */
} imb_capabilities_payload;

typedef struct imb_create_resource_payload {
    uint64_t size;
    uint32_t kind;
    uint32_t options;
} imb_create_resource_payload;

typedef struct imb_create_image_resource_payload {
    uint64_t size;
    uint32_t kind;
    uint32_t options;
    uint32_t width;
    uint32_t height;
    uint32_t format;
    uint32_t reserved;
} imb_create_image_resource_payload;

typedef struct imb_create_sparse_image_resource_payload {
    uint64_t size;
    uint32_t kind;
    uint32_t options;
    uint32_t width;
    uint32_t height;
    uint32_t format;
    uint32_t mip_levels;
    uint32_t array_layers;
    uint32_t sample_count;
    uint32_t texture_type;
    uint32_t reserved;
} imb_create_sparse_image_resource_payload;

typedef struct imb_query_sparse_image_properties_payload {
    uint32_t format;
    uint32_t texture_type;
    uint32_t sample_count;
    uint32_t reserved;
} imb_query_sparse_image_properties_payload;

typedef struct imb_sparse_image_properties_payload {
    uint32_t tile_width;
    uint32_t tile_height;
    uint32_t tile_depth;
    uint32_t reserved;
    uint64_t tile_size_bytes;
} imb_sparse_image_properties_payload;

typedef struct imb_update_sparse_image_mapping_payload {
    uint32_t mode;
    uint32_t mip_level;
    uint32_t slice;
    uint32_t reserved;
    uint32_t tile_x;
    uint32_t tile_y;
    uint32_t tile_z;
    uint32_t tile_width;
    uint32_t tile_height;
    uint32_t tile_depth;
} imb_update_sparse_image_mapping_payload;

typedef struct imb_create_compute_pipeline_payload {
    uint32_t spirv_length;
    uint32_t entry_point_length;
    uint32_t flags;
    uint32_t reserved;
    /* spirv_length bytes followed by entry_point_length bytes of UTF-8 */
} imb_create_compute_pipeline_payload;

typedef struct imb_create_acceleration_structure_payload {
    uint64_t requested_size;
    uint32_t type;
    uint32_t flags;
} imb_create_acceleration_structure_payload;

typedef struct imb_build_primitive_acceleration_structure_payload {
    uint32_t geometry_count;
    uint32_t build_flags;
    uint32_t mode;
    uint32_t reserved;
    /* geometry_count imb_primitive_acceleration_structure_geometry_payload records follow */
} imb_build_primitive_acceleration_structure_payload;

typedef struct imb_primitive_acceleration_structure_geometry_payload {
    uint32_t kind;
    uint32_t flags;
    uint64_t data_resource_id;
    uint64_t data_offset;
    uint32_t primitive_count;
    uint32_t stride;
    uint64_t index_resource_id;
    uint64_t index_offset;
    uint32_t index_type;
    uint32_t vertex_format;
    uint64_t transform_resource_id;
    uint64_t transform_offset;
} imb_primitive_acceleration_structure_geometry_payload;

typedef struct imb_build_instance_acceleration_structure_payload {
    uint32_t instance_count;
    uint32_t build_flags;
    uint32_t mode;
    uint32_t reserved;
    /* instance_count imb_acceleration_structure_instance_payload records follow */
} imb_build_instance_acceleration_structure_payload;

typedef struct imb_acceleration_structure_instance_payload {
    float transformation_matrix[12]; /* Vulkan row-major 3x4 affine transform */
    uint32_t options;
    uint32_t mask;
    uint32_t intersection_function_table_offset;
    uint32_t user_id;
    uint64_t acceleration_structure_resource_id;
    uint64_t reserved;
} imb_acceleration_structure_instance_payload;

typedef struct imb_submit_command_payload {
    uint16_t command;
    uint16_t reserved16;
    uint32_t reserved32;
} imb_submit_command_payload;

typedef struct imb_add_u32_command_payload {
    uint16_t command;
    uint16_t reserved16;
    uint32_t reserved32;
    uint32_t element_count;
    uint32_t addend;
} imb_add_u32_command_payload;

typedef struct imb_draw_triangle_command_payload {
    uint16_t command;
    uint16_t reserved16;
    uint32_t reserved32;
    uint32_t clear_rgba8;
    uint32_t reserved_color;
} imb_draw_triangle_command_payload;

typedef struct imb_draw_indexed_ui_command_payload {
    uint16_t command;
    uint16_t reserved16;
    uint32_t reserved32;
    uint64_t vertex_buffer_id;
    uint64_t index_buffer_id;
    uint64_t vertex_buffer_offset;
    uint64_t index_buffer_offset;
    uint32_t width;
    uint32_t height;
    uint32_t clear_rgba8;
    uint32_t draw_count;
    /* draw_count imb_ui_draw_payload records follow */
} imb_draw_indexed_ui_command_payload;

typedef struct imb_trace_rays_command_payload {
    uint16_t command;
    uint16_t reserved16;
    uint32_t reserved32;
    uint64_t acceleration_structure_id;
    uint32_t width;
    uint32_t height;
    uint32_t miss_rgba8;
    uint32_t hit_rgba8;
    uint32_t options;
    uint32_t reserved1;
    float camera_position[3];
    float camera_forward[3];
    float camera_up[3];
    float vertical_fov_radians;
    float near_distance;
    float far_distance;
    float sphere_light_position[3];
    float sphere_light_color[3];
    float sphere_light_intensity;
    float sphere_light_radius;
    float distant_light_direction[3];
    float distant_light_color[3];
    float distant_light_intensity;
    float distant_light_angle_degrees;
    float dome_light_color[3];
    float dome_light_intensity;
} imb_trace_rays_command_payload;

typedef struct imb_dispatch_compute_command_payload {
    uint16_t command;
    uint16_t reserved16;
    uint32_t reserved32;
    uint32_t group_count_x;
    uint32_t group_count_y;
    uint32_t group_count_z;
    uint32_t binding_count;
    uint32_t push_constant_length;
    uint32_t reserved1;
    /* binding_count records followed by push_constant_length raw bytes */
} imb_dispatch_compute_command_payload;

typedef struct imb_compute_binding_payload {
    uint32_t descriptor_set;
    uint32_t binding;
    uint32_t array_element;
    uint32_t kind;
    /* VkFormat for texel buffers; packed sampler options for sampler kind. */
    uint32_t format;
    uint32_t reserved;
    uint64_t resource_id;
    uint64_t offset;
    uint64_t length;
} imb_compute_binding_payload;

typedef struct imb_ui_draw_payload {
    uint64_t texture_id;
    uint32_t index_count;
    uint32_t first_index;
    int32_t vertex_offset;
    uint32_t scissor_x;
    uint32_t scissor_y;
    uint32_t scissor_width;
    uint32_t scissor_height;
    uint32_t reserved;
} imb_ui_draw_payload;

typedef struct imb_write_resource_payload {
    uint64_t offset;
    uint32_t data_length;
    uint32_t reserved;
    /* data_length bytes follow */
} imb_write_resource_payload;

typedef struct imb_read_resource_payload {
    uint64_t offset;
    uint64_t data_length;
} imb_read_resource_payload;

typedef struct imb_wait_fence_reply_payload {
    uint32_t signaled;
    uint32_t reserved;
} imb_wait_fence_reply_payload;

typedef struct imb_error_payload {
    uint32_t code;
    uint32_t message_length;
    /* message_length bytes of UTF-8 follow */
} imb_error_payload;

#pragma pack(pop)

#ifdef __cplusplus
static_assert(sizeof(imb_message_header) == IMB_PROTOCOL_HEADER_SIZE, "IMB header ABI changed");
static_assert(sizeof(imb_hello_payload) == 8, "IMB HELLO ABI changed");
static_assert(sizeof(imb_hello_reply_payload) == 8, "IMB HELLO_REPLY ABI changed");
static_assert(sizeof(imb_capabilities_payload) == 24, "IMB capabilities ABI changed");
static_assert(sizeof(imb_create_resource_payload) == 16, "IMB resource ABI changed");
static_assert(sizeof(imb_create_image_resource_payload) == 32, "IMB image resource ABI changed");
static_assert(sizeof(imb_create_sparse_image_resource_payload) == 48, "IMB sparse image resource ABI changed");
static_assert(sizeof(imb_query_sparse_image_properties_payload) == 16, "IMB sparse image query ABI changed");
static_assert(sizeof(imb_sparse_image_properties_payload) == 24, "IMB sparse image properties ABI changed");
static_assert(sizeof(imb_update_sparse_image_mapping_payload) == 40, "IMB sparse image mapping ABI changed");
static_assert(sizeof(imb_create_compute_pipeline_payload) == 16, "IMB compute pipeline ABI changed");
static_assert(sizeof(imb_create_acceleration_structure_payload) == 16, "IMB acceleration structure ABI changed");
static_assert(sizeof(imb_build_primitive_acceleration_structure_payload) == 16, "IMB acceleration structure build ABI changed");
static_assert(sizeof(imb_primitive_acceleration_structure_geometry_payload) == 72, "IMB acceleration structure geometry ABI changed");
static_assert(sizeof(imb_build_instance_acceleration_structure_payload) == 16, "IMB instance acceleration structure build ABI changed");
static_assert(sizeof(imb_acceleration_structure_instance_payload) == 80, "IMB acceleration structure instance ABI changed");
static_assert(sizeof(imb_submit_command_payload) == 8, "IMB command ABI changed");
static_assert(sizeof(imb_add_u32_command_payload) == 16, "IMB ADD_U32 ABI changed");
static_assert(sizeof(imb_draw_triangle_command_payload) == 16, "IMB DRAW_TRIANGLE ABI changed");
static_assert(sizeof(imb_draw_indexed_ui_command_payload) == 56, "IMB DRAW_INDEXED_UI ABI changed");
static_assert(sizeof(imb_trace_rays_command_payload) == 168, "IMB TRACE_RAYS ABI changed");
static_assert(sizeof(imb_dispatch_compute_command_payload) == 32, "IMB DISPATCH_COMPUTE ABI changed");
static_assert(sizeof(imb_compute_binding_payload) == 48, "IMB compute binding ABI changed");
static_assert(sizeof(imb_ui_draw_payload) == 40, "IMB UI draw ABI changed");
static_assert(sizeof(imb_write_resource_payload) == 16, "IMB write ABI changed");
static_assert(sizeof(imb_read_resource_payload) == 16, "IMB read ABI changed");
static_assert(sizeof(imb_wait_fence_reply_payload) == 8, "IMB fence ABI changed");
static_assert(sizeof(imb_error_payload) == 8, "IMB error ABI changed");
#else
_Static_assert(sizeof(imb_message_header) == IMB_PROTOCOL_HEADER_SIZE, "IMB header ABI changed");
#endif

#ifdef __cplusplus
}
#endif

#endif /* IMB_PROTOCOL_H */
