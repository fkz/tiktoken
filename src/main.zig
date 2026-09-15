const std = @import("std");
const tokenizer = @import("generate_tokenizer");
const gguf = @import("gguf");

const usage =
    \\Usage:
    \\  tiktoken generate_tokenizer generate <model.gguf|merges.txt> [tokens]
    \\  tiktoken generate_tokenizer extract-merges <model.gguf> [merges.txt]
    \\  tiktoken generate_tokenizer tokenize
    \\  tiktoken generate_tokenizer tokenize-only
    \\  tiktoken generate_tokenizer [hash-search-count] [merges.txt]
    \\  tiktoken gguf <model.gguf> <token-id>...
    \\  tiktoken continue <model.gguf> [text] (reads stdin if omitted; creates ./tokens if missing)
    \\
;

pub fn main(init: std.process.Init) !void {
    var args = init.minimal.args.iterate();
    _ = args.skip();
    const command = args.next() orelse return printUsage(init.io);
    if (std.mem.eql(u8, command, "--help") or std.mem.eql(u8, command, "-h")) {
        return printUsage(init.io);
    }
    if (std.mem.eql(u8, command, "generate_tokenizer")) return tokenizer.run(init, args);
    if (std.mem.eql(u8, command, "gguf")) return gguf.run(init, args);
    if (std.mem.eql(u8, command, "continue")) return gguf.runText(init, args, tokenizer);
    try printUsage(init.io);
    return error.UnknownCommand;
}

fn printUsage(io: std.Io) !void {
    var buffer: [1024]u8 = undefined;
    var writer = std.Io.File.stdout().writer(io, &buffer);
    try writer.interface.writeAll(usage);
    try writer.flush();
}
