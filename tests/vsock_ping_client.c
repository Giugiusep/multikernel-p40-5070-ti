#include <linux/vm_sockets.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

#define SO_VM_SOCKETS_TRANSPORT 9
#define VSOCK_TRANSPORT_MULTIKERNEL 1

static int io_full(int fd, void *buf, size_t n, int write_mode) {
    char *p = buf;
    while (n) { ssize_t r = write_mode ? write(fd,p,n) : read(fd,p,n); if (r <= 0) return -1; p += r; n -= (size_t)r; }
    return 0;
}
static uint64_t ns(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC_RAW, &t); return (uint64_t)t.tv_sec*1000000000ull+t.tv_nsec; }
static int send_chunked(int fd, const char *buf, size_t n) {
    while (n) { size_t chunk = n > 32768 ? 32768 : n; if (io_full(fd, (void *)buf, chunk, 1)) return -1; buf += chunk; n -= chunk; usleep(100); }
    return 0;
}
static int connect_mk(uint32_t cid, uint32_t port) {
    int s = socket(AF_VSOCK, SOCK_STREAM, 0); if (s < 0) return -1;
    int transport = VSOCK_TRANSPORT_MULTIKERNEL;
    if (setsockopt(s, AF_VSOCK, SO_VM_SOCKETS_TRANSPORT, &transport, sizeof(transport)) < 0) { close(s); return -2; }
    struct sockaddr_vm a = { .svm_family=AF_VSOCK, .svm_cid=cid, .svm_port=port };
    if (connect(s, (struct sockaddr *)&a, sizeof(a)) < 0) { close(s); return -3; }
    return s;
}
int main(int argc, char **argv) {
    uint32_t cid=argc>1?strtoul(argv[1],NULL,10):1, port=argc>2?strtoul(argv[2],NULL,10):6000;
    int s=connect_mk(cid,port); if(s<0){fprintf(stderr,"connect_mk=%d\n",s);return 1;}
    uint32_t n=4, out=0; uint64_t t=ns(); io_full(s,&n,4,1); io_full(s,"ping",4,1); if(io_full(s,&out,4,0)||out!=4||io_full(s,&(char[4]){0},4,0)){return 2;} printf("PING_PONG_OK rtt_us=%.3f\n",(ns()-t)/1000.0); close(s);
    const size_t sizes[]={1024,8192,65536,1048576,16777216};
    for(size_t z=0;z<sizeof(sizes)/sizeof(sizes[0]);z++){size_t len=sizes[z];char *b=malloc(len),*r=malloc(len);memset(b,0x5a,len);s=connect_mk(cid,port);if(s<0){fprintf(stderr,"size=%zu connect_mk=%d\n",len,s);return 3;}n=(uint32_t)len;t=ns();if(io_full(s,&n,4,1)||send_chunked(s,b,len)||io_full(s,&out,4,0)||out!=n||io_full(s,r,len,0)){fprintf(stderr,"size=%zu I/O failure out=%u expected=%u\n",len,out,n);return 4;}size_t bad=0;while(bad<len&&b[bad]==r[bad])bad++;if(bad!=len){fprintf(stderr,"size=%zu data mismatch at=%zu got=%02x expected=%02x\n",len,bad,(unsigned char)r[bad],(unsigned char)b[bad]);return 4;}double sec=(ns()-t)/1e9;printf("SIZE=%zu RTT_US=%.3f BW_MiB_s=%.3f\n",len,sec*1e6,(double)len/(1024*1024)/sec);close(s);free(b);free(r);}return 0;
}
