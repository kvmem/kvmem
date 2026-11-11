"""Run the native sequence-buffer lookup against adjacent physical allocations."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from test_attention_rows import function

ROOT = Path(__file__).resolve().parents[2]

HARNESS = r'''
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <vector>
struct Engine {
    struct Buf { void* p; size_t bytes; int per; size_t capacity; };
    struct Seq { std::vector<Buf> bufs; };
    struct Options { int max_context; } opt_;
    int bound_=0, physical=0;
    std::vector<Buf> bufs_;
    std::vector<Seq> seqs_{2};
    int cache_capacity() const { return physical; }
    void* seq_buf(int s,const void* p) const;
};
@FUNCTION@
int main() { try {
    int checks=0;
    for(int logical:{4096,8192})for(int physical:{2560,3072,4096}) {
        Engine e;e.opt_.max_context=logical;e.physical=physical;
        const size_t widths[]={512,32,512,32,96};
        size_t total=0;for(size_t n:widths)total+=n*physical;
        const size_t foreign_stride=512*physical+64;
        std::vector<uint8_t> arena(total+5*foreign_stride+logical*1024);
        size_t offset=0;
        for(size_t n:widths) {
            // Independent allocations are adjacent inside one CPU arena. Lookup
            // must preserve allocation identity, even if logical ranges overlap.
            e.bufs_.push_back({arena.data()+offset,n,1,n*physical});
            // Foreign cudaMalloc allocations need not preserve the source's
            // address order or spacing. Equal layouts would hide a wrong match.
            const size_t target=total+(4-e.seqs_[1].bufs.size())*foreign_stride;
            e.seqs_[1].bufs.push_back({arena.data()+target,n,1,n*physical});
            offset+=n*physical;
        }
        for(size_t i=0;i<e.bufs_.size();++i)for(size_t delta:{size_t(0),size_t(31),e.bufs_[i].capacity-1}) {
            auto* input=(uint8_t*)e.bufs_[i].p+delta;
            auto* expected=(uint8_t*)e.seqs_[1].bufs[i].p+delta;
            if(e.seq_buf(1,input)!=expected)throw std::runtime_error("foreign sequence selected neighboring allocation");
            if(e.seq_buf(0,input)!=input)throw std::runtime_error("bound sequence changed pointer");
            checks+=2;
        }
        // Fixed-size sequence allocations must retain their own byte extent.
        e.bufs_.back().per=0;e.bufs_.back().bytes=e.bufs_.back().capacity;
        if(e.seq_buf(1,e.bufs_.back().p)!=e.seqs_[1].bufs.back().p)throw std::runtime_error("fixed allocation");
        ++checks;
        bool rejected=false;
        try { e.seq_buf(1,arena.data()+arena.size()-1); }
        catch(const std::logic_error&) { rejected=true; }
        if(!rejected)throw std::runtime_error("outside pointer accepted");
        ++checks;
    }
    std::printf("SEQUENCE_BUFFER_PASS checks=%d\n",checks);return 0;
} catch(const std::exception& e) {std::fprintf(stderr,"%s\n",e.what());return 1;}}
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(os.environ.get("POLYSTRATA_SOURCE", ROOT / "backends/polystrata")))
    args = parser.parse_args()
    body = function((args.source / "src/poly/engine.cpp").read_text(encoding="utf-8"), "void* Engine::seq_buf(int s, const void* p) const")
    compiler = shutil.which(os.environ.get("CXX", "cl" if os.name == "nt" else "c++"))
    if not compiler:
        parser.error("C++ compiler not found; use a developer shell or set CXX")
    msvc = Path(compiler).name.lower() in ("cl", "cl.exe", "clang-cl", "clang-cl.exe")
    with tempfile.TemporaryDirectory(prefix="poly-sequence-buffers-") as directory:
        out = Path(directory)
        source = out / "sequence_buffers.cpp"
        source.write_text(HARNESS.replace("@FUNCTION@", body), encoding="utf-8")
        for metal in (False, True):
            binary = out / ("metal" if metal else "cuda_hip")
            if os.name == "nt":
                binary = binary.with_suffix(".exe")
            if msvc:
                cmd = [compiler, "/nologo", "/std:c++17", "/EHsc", "/O2", "/W4", "/Fo" + str(binary.with_suffix(".obj")), "/Fe" + str(binary), str(source)]
            else:
                cmd = [compiler, "-std=c++17", "-O2", "-Wall", "-Wextra", str(source), "-o", str(binary)]
            if metal:
                cmd.append("/DSTRATA_GPU_METAL" if msvc else "-DSTRATA_GPU_METAL")
            subprocess.run(cmd, cwd=out, check=True)
            subprocess.run([str(binary)], cwd=out, check=True)


if __name__ == "__main__":
    main()
