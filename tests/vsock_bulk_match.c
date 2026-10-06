#define _GNU_SOURCE
#include <errno.h>
#include <linux/vm_sockets.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

#define MK_TRANSPORT_OPT 9
#define MK_TRANSPORT_VALUE 1
#define PORT 6001
#define CHUNK (32 * 1024)

struct request { uint32_t bytes, mode, direction; };
struct stats { uint64_t calls, partials, syscall_ns, sleep_ns, total_ns; };

static uint64_t now_ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC_RAW, &t);
    return (uint64_t)t.tv_sec * 1000000000ULL + (uint64_t)t.tv_nsec;
}

/* mode 0: unpaced; 1: exact llama.cpp 32 KiB + usleep(100) after progress;
 * mode 2: 32 KiB cap only; mode 3: large syscalls plus progress sleeps. */
static int io_full(int fd, void *buffer, size_t len, int sending, int mode,
                   struct stats *s) {
    size_t done = 0;
    uint64_t start = now_ns();
    while (done < len) {
        size_t count = len - done;
        if ((mode == 1 || mode == 2) && count > CHUNK) count = CHUNK;
        uint64_t before = now_ns();
        ssize_t n = sending ? send(fd, (char *)buffer + done, count, 0)
                            : recv(fd, (char *)buffer + done, count, 0);
        s->syscall_ns += now_ns() - before;
        s->calls++;
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) return -1;
        if ((size_t)n < count) s->partials++;
        done += (size_t)n;
        if (done < len && (mode == 1 || mode == 3)) {
            before = now_ns();
            usleep(100);
            s->sleep_ns += now_ns() - before;
        }
    }
    s->total_ns = now_ns() - start;
    return 0;
}

static int mk_socket(void) {
    int fd = socket(AF_VSOCK, SOCK_STREAM, 0);
    int transport = MK_TRANSPORT_VALUE;
    if (fd < 0 || setsockopt(fd, AF_VSOCK, MK_TRANSPORT_OPT,
                             &transport, sizeof(transport)) < 0) {
        perror("socket/setsockopt");
        return -1;
    }
    return fd;
}

static int server(void) {
    int listen_fd = mk_socket();
    struct sockaddr_vm addr = {.svm_family = AF_VSOCK,
                               .svm_cid = VMADDR_CID_ANY, .svm_port = PORT};
    if (listen_fd < 0 || bind(listen_fd, (void *)&addr, sizeof(addr)) ||
        listen(listen_fd, 8)) { perror("bind/listen"); return 1; }
    puts("VSOCK_BULK_MATCH_READY"); fflush(stdout);
    for (;;) {
        int fd = accept(listen_fd, NULL, NULL);
        if (fd < 0) { perror("accept"); continue; }
        struct request req;
        struct stats ignored = {0}, measured = {0};
        if (io_full(fd, &req, sizeof(req), 0, 0, &ignored) ||
            req.bytes > 16 * 1024 * 1024 || req.mode > 3 || req.direction > 1) {
            close(fd); continue;
        }
        char *data = malloc(req.bytes);
        if (!data) { close(fd); continue; }
        memset(data, 0x5a, req.bytes);
        if (req.direction == 0) {
            if (io_full(fd, data, req.bytes, 0, req.mode, &measured)) goto end;
            for (uint32_t j = 0; j < req.bytes; ++j) {
                if ((unsigned char)data[j] != 0x5a) {
                    fprintf(stderr, "server mismatch at %u\n", j);
                    goto end;
                }
            }
        } else {
            if (io_full(fd, data, req.bytes, 1, req.mode, &measured)) goto end;
        }
        if (io_full(fd, &measured, sizeof(measured), 1, 0, &ignored)) goto end;
end:
        free(data); close(fd);
    }
}

static int client(uint32_t cid) {
    const uint32_t sizes[] = {512 * 1024, 1024 * 1024, 16 * 1024 * 1024};
    printf("direction,mode,bytes,client_ms,client_syscall_ms,client_sleep_ms,"
           "client_calls,client_partials,server_ms,server_syscall_ms,"
           "server_sleep_ms,server_calls,server_partials,MiB_s\n");
    for (int direction = 0; direction < 2; ++direction)
    for (int mode = 0; mode < 4; ++mode)
    for (unsigned i = 0; i < sizeof(sizes)/sizeof(sizes[0]); ++i) {
        int fd = mk_socket();
        struct sockaddr_vm addr = {.svm_family = AF_VSOCK,
                                   .svm_cid = cid, .svm_port = PORT};
        if (fd < 0 || connect(fd, (void *)&addr, sizeof(addr))) {
            perror("connect"); return 2;
        }
        struct request req = {sizes[i], (uint32_t)mode, (uint32_t)direction};
        struct stats ignored = {0}, client_stats = {0}, server_stats = {0};
        char *data = malloc(req.bytes);
        if (!data) return 3;
        memset(data, 0x5a, req.bytes);
        if (io_full(fd, &req, sizeof(req), 1, 0, &ignored) ||
            io_full(fd, data, req.bytes, direction == 0, mode, &client_stats) ||
            io_full(fd, &server_stats, sizeof(server_stats), 0, 0, &ignored)) {
            perror("data/stats"); return 4;
        }
        for (uint32_t j = 0; j < req.bytes; ++j) {
            if (data[j] != 0x5a) { fprintf(stderr, "mismatch at %u\n", j); return 5; }
        }
        double sec = client_stats.total_ns / 1e9;
        printf("%s,%d,%u,%.3f,%.3f,%.3f,%llu,%llu,%.3f,%.3f,%.3f,%llu,%llu,%.3f\n",
               direction ? "P40-to-primary" : "primary-to-P40", mode, req.bytes,
               client_stats.total_ns / 1e6, client_stats.syscall_ns / 1e6,
               client_stats.sleep_ns / 1e6,
               (unsigned long long)client_stats.calls,
               (unsigned long long)client_stats.partials,
               server_stats.total_ns / 1e6, server_stats.syscall_ns / 1e6,
               server_stats.sleep_ns / 1e6,
               (unsigned long long)server_stats.calls,
               (unsigned long long)server_stats.partials,
               (req.bytes / 1048576.0) / sec);
        fflush(stdout);
        free(data); close(fd);
    }
    return 0;
}

int main(int argc, char **argv) {
    if (argc > 1 && strcmp(argv[1], "server") == 0) return server();
    return client(argc > 1 ? (uint32_t)strtoul(argv[1], NULL, 10) : 1);
}
