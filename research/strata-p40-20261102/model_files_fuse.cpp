#define FUSE_USE_VERSION 31
#include <fuse3/fuse.h>
#include <sys/socket.h>
#include <linux/vm_sockets.h>
#include <unistd.h>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>
#include <cstdio>
#include <fcntl.h>
static int peer=-1;static std::mutex lock;
struct Req{uint32_t op,n;uint64_t off,count;};struct Resp{int32_t status;uint32_t mode;uint64_t size,n;};
static bool io(void*p,size_t n,bool send){auto q=(char*)p;while(n){ssize_t k=send?::send(peer,q,n,0):recv(peer,q,n,0);if(k<0&&errno==EINTR)continue;if(k<=0)return false;q+=k;n-=k;}return true;}
static int call(uint32_t op,const char*path,uint64_t off,uint64_t count,Resp&r,std::vector<char>&data){std::lock_guard<std::mutex>g(lock);Req req{op,(uint32_t)strlen(path),off,count};if(req.n>4096||count>1048576)return -EINVAL;if(!io(&req,sizeof req,true)||!io((void*)path,req.n,true)||!io(&r,sizeof r,false))return -EIO;if(r.n>1048576)return -EIO;data.resize(r.n);if(r.n&&!io(data.data(),r.n,false))return -EIO;return r.status;}
static int getattr_(const char*p,struct stat*st,fuse_file_info*){Resp r;std::vector<char>d;int e=call(1,p,0,0,r,d);if(e)return e;memset(st,0,sizeof *st);st->st_mode=r.mode&~0222;st->st_size=r.size;st->st_nlink=S_ISDIR(r.mode)?2:1;st->st_blksize=131072;st->st_blocks=(r.size+511)/512;st->st_uid=getuid();st->st_gid=getgid();return 0;}
static int open_(const char*p,fuse_file_info*fi){if((fi->flags&O_ACCMODE)!=O_RDONLY)return -EROFS;struct stat st;int e=getattr_(p,&st,fi);if(e)return e;if(!S_ISREG(st.st_mode))return -EISDIR;fi->keep_cache=1;return 0;}
static int read_(const char*p,char*buf,size_t n,off_t off,fuse_file_info*){Resp r;std::vector<char>d;int e=call(3,p,off,n,r,d);if(e)return e;memcpy(buf,d.data(),d.size());return d.size();}
static int readdir_(const char*p,void*buf,fuse_fill_dir_t fill,off_t,fuse_file_info*,fuse_readdir_flags){Resp r;std::vector<char>d;int e=call(2,p,0,0,r,d);if(e)return e;fill(buf,".",nullptr,0,(fuse_fill_dir_flags)0);fill(buf,"..",nullptr,0,(fuse_fill_dir_flags)0);std::string s(d.begin(),d.end());size_t a=0;while(a<s.size()){auto b=s.find('\n',a);if(b==std::string::npos)break;std::string name=s.substr(a,b-a);if(!name.empty())fill(buf,name.c_str(),nullptr,0,(fuse_fill_dir_flags)0);a=b+1;}return 0;}
static void*init_(fuse_conn_info*c,fuse_config*cfg){cfg->kernel_cache=1;cfg->attr_timeout=3600;cfg->entry_timeout=3600;c->max_read=131072;c->max_readahead=131072;return nullptr;}
int main(int argc,char**argv){peer=socket(AF_VSOCK,SOCK_STREAM,0);int type=1;if(peer<0||setsockopt(peer,AF_VSOCK,9,&type,sizeof type)){perror("mkvsock");return 1;}sockaddr_vm a{};a.svm_family=AF_VSOCK;a.svm_cid=0;a.svm_port=6004;if(connect(peer,(sockaddr*)&a,sizeof a)){perror("connect");return 1;}fuse_operations ops{};ops.getattr=getattr_;ops.open=open_;ops.read=read_;ops.readdir=readdir_;ops.init=init_;return fuse_main(argc,argv,&ops,nullptr);}
