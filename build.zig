const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const thread_count = b.option(usize, "thread-count", "Total GGUF compute threads, including the calling thread (default: 1)") orelse 1;
    if (thread_count == 0) @panic("-Dthread-count must be at least 1");
    const prefetch = b.option(usize, "prefetch", "GGUF weight prefetch distance in 4 KiB blocks (0 disables, default: 0)") orelse 0;
    const gguf_options = b.addOptions();
    gguf_options.addOption(usize, "thread_count", thread_count);
    gguf_options.addOption(usize, "prefetch", prefetch);
    const tokenizer = b.createModule(.{
        .root_source_file = b.path("tools/generate_tokenizer.zig"),
        .target = target,
        .optimize = optimize,
    });
    const gguf = b.createModule(.{
        .root_source_file = b.path("tools/gguf.zig"),
        .target = target,
        .optimize = optimize,
    });
    gguf.addOptions("build_options", gguf_options);
    const exe = b.addExecutable(.{
        .name = "tiktoken",
        // The GGUF kernels use vector inline assembly unsupported by the native backend.
        .use_llvm = true,
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
            .strip = false,
            .imports = &.{
                .{ .name = "generate_tokenizer", .module = tokenizer },
                .{ .name = "gguf", .module = gguf },
            },
        }),
    });
    b.installArtifact(exe);
    const run = b.addRunArtifact(exe);
    if (b.args) |args| run.addArgs(args);
    b.step("run", "Run the program").dependOn(&run.step);

    const test_step = b.step("test", "Run self-contained tests");
    const parsing_tests = b.addTest(.{ .root_module = tokenizer, .filters = &.{"parsing"} });
    test_step.dependOn(&b.addRunArtifact(parsing_tests).step);
    const gguf_tests = b.addTest(.{ .root_module = gguf });
    test_step.dependOn(&b.addRunArtifact(gguf_tests).step);
    const merges_tests = b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("tools/gguf_merges.zig"),
        .target = target,
        .optimize = optimize,
    }) });
    test_step.dependOn(&b.addRunArtifact(merges_tests).step);
}
