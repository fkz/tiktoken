// Vulkan compute streaming reads. Setup and readback are excluded from timing.
#define _GNU_SOURCE
#include <vulkan/vulkan.h>
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "stream-timing.h"

#define VK(call) do { VkResult r = (call); if (r != VK_SUCCESS) { fprintf(stderr, "%s: Vulkan error %d\n", #call, r); exit(1); } } while (0)
static VkDevice device;
static VkPhysicalDevice physical;
struct buffer { VkBuffer handle; VkDeviceMemory memory; VkDeviceSize size; void *mapped; };
static double now(void) {
    struct timespec t;
    if (clock_gettime(CLOCK_MONOTONIC, &t)) { perror("clock_gettime"); exit(1); }
    return t.tv_sec + t.tv_nsec * 1e-9;
}
static struct buffer make_buffer(VkDeviceSize size) {
    struct buffer b = {.size = size};
    VkBufferCreateInfo info = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = size,
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
    VK(vkCreateBuffer(device, &info, NULL, &b.handle));
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(device, b.handle, &req);
    VkPhysicalDeviceMemoryProperties props;
    vkGetPhysicalDeviceMemoryProperties(physical, &props);
    uint32_t index = UINT32_MAX;
    // UMA only: use memory accessible to both this integrated GPU and the host.
    VkMemoryPropertyFlags wanted = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    for (uint32_t i = 0; i < props.memoryTypeCount; i++)
        if ((req.memoryTypeBits & (1u << i)) && (props.memoryTypes[i].propertyFlags & wanted) == wanted) { index = i; break; }
    if (index == UINT32_MAX) { fprintf(stderr, "No coherent host-visible device-local memory\n"); exit(1); }
    VkMemoryAllocateInfo alloc = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = req.size, .memoryTypeIndex = index};
    VK(vkAllocateMemory(device, &alloc, NULL, &b.memory));
    VK(vkBindBufferMemory(device, b.handle, b.memory, 0));
    VK(vkMapMemory(device, b.memory, 0, size, 0, &b.mapped));
    return b;
}
static void free_buffer(struct buffer b) {
    vkUnmapMemory(device, b.memory); vkDestroyBuffer(device, b.handle, NULL); vkFreeMemory(device, b.memory, NULL);
}

int main(int argc, char **argv) {
    if (argc != 2) { fprintf(stderr, "Usage: %s SHADER.spv\n", argv[0]); return 2; }
    double duration = stream_seconds();
    if (!duration) duration = 3;
    VkApplicationInfo app = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .pApplicationName = "tik GPU streaming", .apiVersion = VK_API_VERSION_1_1};
    VkInstanceCreateInfo ii = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &app};
    VkInstance instance;
    VK(vkCreateInstance(&ii, NULL, &instance));
    uint32_t count = 0;
    VK(vkEnumeratePhysicalDevices(instance, &count, NULL));
    VkPhysicalDevice *devices = calloc(count, sizeof(*devices));
    if (!devices) return 1;
    VK(vkEnumeratePhysicalDevices(instance, &count, devices));
    VkPhysicalDeviceProperties props;
    for (uint32_t i = 0; i < count; i++) {
        vkGetPhysicalDeviceProperties(devices[i], &props);
        if (props.vendorID == 0x1002 && props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) { physical = devices[i]; break; }
    }
    free(devices);
    if (!physical) { fprintf(stderr, "No AMD integrated hardware GPU found; CPU Vulkan drivers are refused.\n"); return 1; }
    vkGetPhysicalDeviceProperties(physical, &props);
    fprintf(stderr, "GPU: %s; driver version: %u\n", props.deviceName, props.driverVersion);
    const VkDeviceSize bytes = (VkDeviceSize)1 << 30, output_bytes = 65536 * 16;
    if (props.limits.maxStorageBufferRange < bytes || props.limits.maxComputeWorkGroupInvocations < 256 || props.limits.maxComputeWorkGroupSize[0] < 256) {
        fprintf(stderr, "GPU limits do not support this benchmark\n"); return 1;
    }
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &count, NULL);
    VkQueueFamilyProperties *families = calloc(count, sizeof(*families));
    if (!families) return 1;
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &count, families);
    uint32_t family = UINT32_MAX, valid_bits = 0;
    for (uint32_t i = 0; i < count; i++) {
        if ((families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) && families[i].timestampValidBits) {
            family = i; valid_bits = families[i].timestampValidBits;
            if (!(families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) break;
        }
    }
    free(families);
    if (family == UINT32_MAX) { fprintf(stderr, "No compute queue with timestamps\n"); return 1; }
    float priority = 1;
    VkDeviceQueueCreateInfo qi = {.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueFamilyIndex = family, .queueCount = 1, .pQueuePriorities = &priority};
    VkDeviceCreateInfo di = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .queueCreateInfoCount = 1, .pQueueCreateInfos = &qi};
    VK(vkCreateDevice(physical, &di, NULL, &device));
    VkQueue queue; vkGetDeviceQueue(device, family, 0, &queue);
    struct buffer input = make_buffer(bytes), output = make_buffer(output_bytes);
    // Deterministic varied values avoid benchmarking a zero-filled/compressible buffer.
    uint32_t *words = input.mapped;
    for (uint32_t i = 0; i < bytes / 4; i++) words[i] = i * 1664525u + 1013904223u;
    memset(output.mapped, 0, output_bytes);
    uint32_t *expected = calloc(output_bytes / 4, 4);
    if (!expected) return 1;
    for (uint32_t i = 0; i < bytes / 4; i++) expected[i % (output_bytes / 4)] += words[i];

    FILE *file = fopen(argv[1], "rb");
    if (!file) { perror("shader"); return 1; }
    fseek(file, 0, SEEK_END); long length = ftell(file); rewind(file);
    if (length <= 0 || length % 4) return 1;
    uint32_t *code = malloc(length);
    if (!code || fread(code, 1, length, file) != (size_t)length) return 1;
    fclose(file);
    VkShaderModuleCreateInfo si = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = length, .pCode = code};
    VkShaderModule shader; VK(vkCreateShaderModule(device, &si, NULL, &shader)); free(code);
    VkDescriptorSetLayoutBinding bindings[2];
    for (int i = 0; i < 2; i++) bindings[i] = (VkDescriptorSetLayoutBinding){.binding = i, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT};
    VkDescriptorSetLayoutCreateInfo li = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .bindingCount = 2, .pBindings = bindings};
    VkDescriptorSetLayout layout; VK(vkCreateDescriptorSetLayout(device, &li, NULL, &layout));
    VkPipelineLayoutCreateInfo pli = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1, .pSetLayouts = &layout};
    VkPipelineLayout pipeline_layout; VK(vkCreatePipelineLayout(device, &pli, NULL, &pipeline_layout));
    VkComputePipelineCreateInfo pi = {.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, .layout = pipeline_layout,
        .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = shader, .pName = "main"}};
    VkPipeline pipeline; VK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pi, NULL, &pipeline));
    VkDescriptorPoolSize pool_size = {.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 2};
    VkDescriptorPoolCreateInfo dpi = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .maxSets = 1, .poolSizeCount = 1, .pPoolSizes = &pool_size};
    VkDescriptorPool pool; VK(vkCreateDescriptorPool(device, &dpi, NULL, &pool));
    VkDescriptorSetAllocateInfo dai = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, .descriptorPool = pool, .descriptorSetCount = 1, .pSetLayouts = &layout};
    VkDescriptorSet set; VK(vkAllocateDescriptorSets(device, &dai, &set));
    VkDescriptorBufferInfo bis[2] = {{input.handle, 0, bytes}, {output.handle, 0, output_bytes}};
    VkWriteDescriptorSet writes[2];
    for (int i = 0; i < 2; i++) writes[i] = (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = set, .dstBinding = i, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &bis[i]};
    vkUpdateDescriptorSets(device, 2, writes, 0, NULL);
    VkQueryPoolCreateInfo qpi = {.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, .queryType = VK_QUERY_TYPE_TIMESTAMP, .queryCount = 2};
    VkQueryPool queries; VK(vkCreateQueryPool(device, &qpi, NULL, &queries));
    VkCommandPoolCreateInfo cpi = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, .queueFamilyIndex = family};
    VkCommandPool command_pool; VK(vkCreateCommandPool(device, &cpi, NULL, &command_pool));
    VkCommandBufferAllocateInfo cai = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = command_pool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1};
    VkCommandBuffer cmd; VK(vkAllocateCommandBuffers(device, &cai, &cmd));
    VkCommandBufferBeginInfo begin = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    VK(vkBeginCommandBuffer(cmd, &begin));
    vkCmdResetQueryPool(cmd, queries, 0, 2);
    // Also order output writes against the previous submission's output writes.
    VkMemoryBarrier host_in = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER, .srcAccessMask = VK_ACCESS_HOST_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT, .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT};
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &host_in, 0, NULL, 0, NULL);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout, 0, 1, &set, 0, NULL);
    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queries, 0);
    vkCmdDispatch(cmd, 256, 1, 1);
    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queries, 1);
    VkMemoryBarrier host_out = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER, .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT, .dstAccessMask = VK_ACCESS_HOST_READ_BIT};
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &host_out, 0, NULL, 0, NULL);
    VK(vkEndCommandBuffer(cmd));
    VkFenceCreateInfo fi = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkFence fence; VK(vkCreateFence(device, &fi, NULL, &fence));
    VkSubmitInfo submit = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cmd};
    // Warm the pipeline and validate before coordinated measurement starts.
    VK(vkQueueSubmit(queue, 1, &submit, fence));
    VK(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX));
    if (memcmp(output.mapped, expected, output_bytes)) { fprintf(stderr, "GPU checksum failed\n"); return 1; }
    stream_wait_start();
    double before = now(), gpu_seconds = 0;
    uint64_t passes = 0;
    do {
        VK(vkResetFences(device, 1, &fence));
        VK(vkQueueSubmit(queue, 1, &submit, fence));
        VK(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX));
        uint64_t stamps[2];
        VK(vkGetQueryPoolResults(device, queries, 0, 2, sizeof(stamps), stamps, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
        uint64_t mask = valid_bits == 64 ? UINT64_MAX : (UINT64_C(1) << valid_bits) - 1;
        gpu_seconds += ((stamps[1] - stamps[0]) & mask) * props.limits.timestampPeriod * 1e-9;
        passes++;
    } while (now() - before < duration);
    double after = now();
    if (memcmp(output.mapped, expected, output_bytes)) { fprintf(stderr, "GPU checksum failed\n"); return 1; }
    printf("{\"buffer_bytes\":%llu,\"read_bytes\":%llu,\"passes\":%llu,\"seconds\":%.9f,\"start_s\":%.9f,\"end_s\":%.9f,\"GB_s\":%.6f,\"gpu_GB_s\":%.6f}\n",
        (unsigned long long)bytes, (unsigned long long)(bytes * passes), (unsigned long long)passes,
        after - before, before, after, (double)bytes * passes / (after - before) / 1e9, (double)bytes * passes / gpu_seconds / 1e9);
    free(expected);
    vkDestroyFence(device, fence, NULL); vkDestroyCommandPool(device, command_pool, NULL);
    vkDestroyQueryPool(device, queries, NULL); vkDestroyDescriptorPool(device, pool, NULL);
    vkDestroyPipeline(device, pipeline, NULL); vkDestroyPipelineLayout(device, pipeline_layout, NULL);
    vkDestroyDescriptorSetLayout(device, layout, NULL); vkDestroyShaderModule(device, shader, NULL);
    free_buffer(output); free_buffer(input); vkDestroyDevice(device, NULL); vkDestroyInstance(instance, NULL);
    return 0;
}
