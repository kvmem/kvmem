// Linux-only, opt-in LD_PRELOAD fixture. It never changes the backing file.
// An armed marker injects exactly one EIO into a spill read owned by this run.
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <limits.h>
#include <time.h>
#include <unistd.h>

namespace {
bool claim(int fd,const char* marker_name) {
    const char* directory=std::getenv("KVMEM_TEST_SPILL_DIRECTORY");
    const char* marker=std::getenv(marker_name);
    if(!directory || directory[0]!='/' || !marker || marker[0]!='/')return false;
    char link[64],path[PATH_MAX+1];
    const int written=std::snprintf(link,sizeof(link),"/proc/self/fd/%d",fd);
    if(written<0 || size_t(written)>=sizeof(link))return false;
    const auto n=::readlink(link,path,sizeof(path)-1);
    if(n<=0 || size_t(n)>=sizeof(path)-1)return false;
    path[n]=0;
    const size_t size=std::strlen(directory);
    if(std::strncmp(path,directory,size) || path[size]!='/' ||
        std::strncmp(path+size+1,"kvmem-spill-",12))return false;
    // unlink is the atomic one-shot claim, including with concurrent I/O workers.
    return ::unlink(marker)==0;
}
bool inject(int fd) {
    if(!claim(fd,"KVMEM_TEST_READ_FAULT_MARKER"))return false;
    const char notice[]="KVMEM_TEST_READ_FAULT injected EIO in owned spill read\n";
    (void)::write(STDERR_FILENO,notice,sizeof(notice)-1);
    return true;
}
bool inject_write(int fd) {
    if(!claim(fd,"KVMEM_TEST_WRITE_FAULT_MARKER"))return false;
    const char notice[]="KVMEM_TEST_WRITE_FAULT injected EIO in owned spill write\n";
    (void)::write(STDERR_FILENO,notice,sizeof(notice)-1);
    return true;
}
void delay_write(int fd) {
    if(!claim(fd,"KVMEM_TEST_WRITE_DELAY_MARKER"))return;
    const char begin[]="KVMEM_TEST_WRITE_DELAY begin\n",end[]="KVMEM_TEST_WRITE_DELAY end\n";
    (void)::write(STDERR_FILENO,begin,sizeof(begin)-1);
    long seconds=5;
    if(const char* value=std::getenv("KVMEM_TEST_WRITE_DELAY_SECONDS")) {
        char* tail=nullptr;const long parsed=std::strtol(value,&tail,10);
        if(tail && !*tail && parsed>=1 && parsed<=60)seconds=parsed;
    }
    timespec remaining{seconds,0};while(::nanosleep(&remaining,&remaining) && errno==EINTR){}
    (void)::write(STDERR_FILENO,end,sizeof(end)-1);
}
void delay_read(int fd) {
    if(!claim(fd,"KVMEM_TEST_READ_DELAY_MARKER"))return;
    const char begin[]="KVMEM_TEST_READ_DELAY begin\n",end[]="KVMEM_TEST_READ_DELAY end\n";
    (void)::write(STDERR_FILENO,begin,sizeof(begin)-1);
    // Test-only bounded override for the real 30-second read timeout path.
    long seconds=5;
    if(const char* value=std::getenv("KVMEM_TEST_READ_DELAY_SECONDS")) {
        char* end=nullptr;const long parsed=std::strtol(value,&end,10);
        if(end && !*end && parsed>=1 && parsed<=60)seconds=parsed;
    }
    timespec remaining{seconds,0};while(::nanosleep(&remaining,&remaining) && errno==EINTR){}
    (void)::write(STDERR_FILENO,end,sizeof(end)-1);
}
}
extern "C" ssize_t pread(int fd,void* bytes,size_t size,off_t offset) {
    using Function=ssize_t(*)(int,void*,size_t,off_t);
    static const auto next=reinterpret_cast<Function>(::dlsym(RTLD_NEXT,"pread"));
    const int previous=errno;delay_read(fd);
    if(inject(fd)){errno=EIO;return -1;}
    if(!next){errno=ENOSYS;return -1;}
    errno=previous;return next(fd,bytes,size,offset);
}
extern "C" ssize_t pread64(int fd,void* bytes,size_t size,off64_t offset) {
    using Function=ssize_t(*)(int,void*,size_t,off64_t);
    static const auto next=reinterpret_cast<Function>(::dlsym(RTLD_NEXT,"pread64"));
    const int previous=errno;delay_read(fd);
    if(inject(fd)){errno=EIO;return -1;}
    if(!next){errno=ENOSYS;return -1;}
    errno=previous;return next(fd,bytes,size,offset);
}
extern "C" ssize_t pwrite(int fd,const void* bytes,size_t size,off_t offset) {
    using Function=ssize_t(*)(int,const void*,size_t,off_t);
    static const auto next=reinterpret_cast<Function>(::dlsym(RTLD_NEXT,"pwrite"));
    const int previous=errno;delay_write(fd);
    if(inject_write(fd)){errno=EIO;return -1;}
    if(!next){errno=ENOSYS;return -1;}
    errno=previous;return next(fd,bytes,size,offset);
}
extern "C" ssize_t pwrite64(int fd,const void* bytes,size_t size,off64_t offset) {
    using Function=ssize_t(*)(int,const void*,size_t,off64_t);
    static const auto next=reinterpret_cast<Function>(::dlsym(RTLD_NEXT,"pwrite64"));
    const int previous=errno;delay_write(fd);
    if(inject_write(fd)){errno=EIO;return -1;}
    if(!next){errno=ENOSYS;return -1;}
    errno=previous;return next(fd,bytes,size,offset);
}
