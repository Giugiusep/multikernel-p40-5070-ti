#include <arpa/inet.h>
#include <errno.h>
#include <linux/vm_sockets.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

#define SO_VM_SOCKETS_TRANSPORT 9
#define VSOCK_TRANSPORT_MULTIKERNEL 1

static int io_full(int fd, void *buf, size_t n, int write_mode) {
    char *p = buf;
    while (n) {
        ssize_t r = write_mode ? write(fd, p, n) : read(fd, p, n);
        if (r <= 0) return -1;
        p += r; n -= (size_t)r;
    }
    return 0;
}

int main(int argc, char **argv) {
    uint32_t port = argc > 1 ? (uint32_t)strtoul(argv[1], NULL, 10) : 6000;
    int s = socket(AF_VSOCK, SOCK_STREAM, 0);
    if (s < 0) { perror("socket"); return 1; }
    int transport = VSOCK_TRANSPORT_MULTIKERNEL;
    if (setsockopt(s, AF_VSOCK, SO_VM_SOCKETS_TRANSPORT, &transport, sizeof(transport)) < 0) {
        perror("setsockopt transport"); return 2;
    }
    struct sockaddr_vm a = { .svm_family = AF_VSOCK, .svm_cid = VMADDR_CID_ANY, .svm_port = port };
    if (bind(s, (struct sockaddr *)&a, sizeof(a)) < 0) { perror("bind"); return 3; }
    if (listen(s, 8) < 0) { perror("listen"); return 4; }
    puts("VSOCK_NATIVE_SERVER_READY"); fflush(stdout);
    for (;;) {
        int c = accept(s, NULL, NULL);
        if (c < 0) { if (errno == EINTR) continue; perror("accept"); return 5; }
        uint32_t len;
        if (io_full(c, &len, sizeof(len), 0) == 0 && len <= 16 * 1024 * 1024) {
            char *buf = malloc(len ? len : 1);
            if (!buf || io_full(c, buf, len, 0) < 0) { free(buf); close(c); continue; }
            if (len == 4 && memcmp(buf, "ping", 4) == 0) {
                uint32_t out = 4; io_full(c, &out, sizeof(out), 1); io_full(c, "pong", 4, 1);
            } else {
                io_full(c, &len, sizeof(len), 1); io_full(c, buf, len, 1);
            }
            free(buf);
        }
        close(c);
    }
}
