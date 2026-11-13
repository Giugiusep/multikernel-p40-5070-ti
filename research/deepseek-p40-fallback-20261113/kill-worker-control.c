#include <sys/socket.h>
#include <linux/vm_sockets.h>
#include <signal.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
int main(int argc,char **argv){if(argc!=2)return 1;int pid=atoi(argv[1]);char path[80],cmd[2048]={0};snprintf(path,sizeof(path),"/proc/%d/cmdline",pid);FILE*f=fopen(path,"rb");if(!f)return 2;size_t n=fread(cmd,1,sizeof(cmd)-1,f);fclose(f);for(size_t i=0;i<n;i++)if(!cmd[i])cmd[i]=' ';if(!strstr(cmd,"/tmp/rxsafe-test/worker/worker-compact"))return 3;int s=socket(AF_VSOCK,SOCK_STREAM,0),v=1;struct sockaddr_vm a={.svm_family=AF_VSOCK,.svm_cid=VMADDR_CID_ANY,.svm_port=6005};if(setsockopt(s,AF_VSOCK,9,&v,4)||bind(s,(void*)&a,sizeof(a))||listen(s,1))return 4;puts("CONTROL_READY");fflush(stdout);int c=accept(s,0,0);unsigned magic=0;if(c<0||recv(c,&magic,4,MSG_WAITALL)!=4||magic!=0x4b494c4c)return 5;int rc=kill(pid,SIGTERM);send(c,&rc,4,MSG_NOSIGNAL);close(c);close(s);return rc!=0;}
