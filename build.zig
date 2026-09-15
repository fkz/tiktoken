const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
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
