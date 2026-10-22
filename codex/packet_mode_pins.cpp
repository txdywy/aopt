#include <algorithm>
#include <array>
#include <fstream>
#include <iostream>
#include <queue>
#include <random>
#include <string>
#include <utility>
#include <vector>

using Edges=std::vector<std::vector<std::pair<int,int>>>;
struct Result {int h;std::vector<int> times,mode;};

int main(int argc,char**argv){
 if(argc<5)return 2;
 std::ifstream in(argv[1]);int n,ne,ng;in>>n>>ne>>ng;
 std::vector<int> engine(n),original(n),gid(n,-1),oldmode(ng);
 for(int i=0;i<n;i++)in>>engine[i]>>original[i];
 Edges pred(n),succ(n);
 for(int z=0;z<ne;z++){int a,b,l;in>>a>>b>>l;pred[b].emplace_back(a,l);succ[a].emplace_back(b,l);}
 std::vector<std::array<int,8>> groups(ng);
 for(int g=0;g<ng;g++){in>>oldmode[g];for(int&i:groups[g]){in>>i;gid[i]=g;}}
 if(!in)return 3;
 const int iterations=std::stoi(argv[3]);std::mt19937_64 rng(std::stoull(argv[4]));
 const double scalar_bias=argc>5?std::stod(argv[5]):0.0;
 std::vector<int> pins(n,-1),pinmode(ng,-1);
 if(argc>6){
  std::ifstream pf(argv[6]);int count;pf>>count;
  for(int i=0;i<count;i++){int j,t;pf>>j>>t;pins[j]=t;}
  pf>>count;for(int i=0;i<count;i++){int g,m;pf>>g>>m;pinmode[g]=m;}
  if(!pf)return 5;
 }
 const int caps[5]={12,6,2,2,1};
 auto run=[&](const std::vector<int>&ori,const std::vector<int>&prev,bool reverse,double noise,int defer)->Result{
  const auto&deps=reverse?succ:pred;const auto&next=reverse?pred:succ;
  int high=*std::max_element(ori.begin(),ori.end());
  std::vector<int> target=pins;
  if(reverse)for(int j=0;j<n;j++)if(pins[j]>=0)target[j]=high-pins[j];
  std::vector<double> priority(n);std::uniform_real_distribution<double>u(-noise,noise);
  for(int j=0;j<n;j++)priority[j]=(target[j]>=0)?-1000000.0+target[j]:
   (reverse?high-ori[j]:ori[j])+u(rng)+(engine[j]==0?scalar_bias:0.0);
  std::vector<int> degree(n),earliest(n),times(n,-1),readycount(ng),partial(ng),mode(ng);
  using Item=std::pair<double,int>;std::priority_queue<Item,std::vector<Item>,std::greater<Item>> ready;
  using Future=std::pair<int,int>;std::priority_queue<Future,std::vector<Future>,std::greater<Future>> future;
  auto addready=[&](int j){if(gid[j]>=0)readycount[gid[j]]++;ready.emplace(priority[j],j);};
  for(int j=0;j<n;j++){degree[j]=deps[j].size();if(!degree[j])addready(j);}
  int t=0,done=0;
  while(done<n && t<2500){
   while(!future.empty()&&future.top().first<=t){int j=future.top().second;future.pop();addready(j);}
   int used[5]={};std::vector<Item>blocked;
   auto issue=[&](int j){times[j]=t;done++;for(auto[q,l]:next[j]){earliest[q]=std::max(earliest[q],t+l);if(!--degree[q]){if(earliest[q]<=t)addready(q);else future.emplace(earliest[q],q);}}};
   while(!ready.empty()){
    auto item=ready.top();ready.pop();auto[pr,j]=item;if(times[j]>=0)continue;int g=gid[j];
    if(target[j]>=0&&target[j]>t){blocked.push_back(item);continue;}
    if(target[j]>=0&&target[j]<t)return {9999,{}, {}};
    bool allow_native=true;
    if(g>=0){
     if(pinmode[g]==0)allow_native=false;
     for(int a:groups[g])if(target[a]>t)allow_native=false;
    }
    if(g>=0&&allow_native&&!partial[g]&&readycount[g]==8&&used[1]<6){
     used[1]++;partial[g]=1;mode[g]=1;for(int a:groups[g])issue(a);continue;
    }
    if(g>=0&&pinmode[g]==1){blocked.push_back(item);continue;}
    if(used[engine[j]]>=caps[engine[j]]){blocked.push_back(item);continue;}
    if(g>=0&&!partial[g]&&prev[g]&&readycount[g]<8&&defer>0&&pr+defer>t){blocked.push_back(item);continue;}
    used[engine[j]]++;if(g>=0)partial[g]=1;issue(j);
   }
   for(auto item:blocked)ready.push(item);
   t++;
  }
  if(done!=n)return {9999,{}, {}};
  if(reverse)for(int&x:times)x=t-1-x;
  return {t,std::move(times),std::move(mode)};
 };
 Result best{9999,{},{}},current{*std::max_element(original.begin(),original.end())+1,original,oldmode};
 const double noises[]={0,.49,1.95,3.95,7.95,15.95};const int defers[]={0,1,2,4};
 for(int it=0;it<iterations;it++){
  if(it%16==15&&!best.times.empty())current=best;
  for(int pass=0;pass<4;pass++){
   auto result=run(current.times,current.mode,(it+pass)%2,pass==0?noises[it%6]:0,defers[it%4]);
   if(result.h==9999)continue;
   current=std::move(result);
   if(current.h<best.h){
    best=current;std::cout<<"iteration "<<it<<" pass "<<pass<<" cycles "<<best.h<<std::endl;
    std::ofstream out(argv[2]);out<<n<<' '<<ng<<'\n';for(int v:best.times)out<<v<<' ';out<<'\n';for(int v:best.mode)out<<v<<' ';out<<'\n';
   }
  }
 }
}
