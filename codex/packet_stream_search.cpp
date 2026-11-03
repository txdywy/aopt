#include <algorithm>
#include <array>
#include <fstream>
#include <iostream>
#include <queue>
#include <random>
#include <utility>
#include <vector>

// Search a projected, fixed-horizon LF calendar. This cannot certify compute
// packing or a score: the resulting stream still needs exact ALU/VALU repair.
int main(int argc,char**argv){
 if(argc<5)return 2;
 std::ifstream in(argv[1]);int n,ne,nf,H;in>>n>>ne>>nf>>H;
 std::vector<int>eng(n),hint(n),lo(n),hi(n),id(n);
 for(int j=0;j<n;j++)in>>eng[j]>>hint[j]>>lo[j]>>hi[j]>>id[j];
 using Edges=std::vector<std::vector<std::pair<int,int>>>;
 Edges pred(n),succ(n);
 for(int z=0;z<ne;z++){int d,j,l;in>>d>>j>>l;pred[j].emplace_back(d,l);succ[d].emplace_back(j,l);}
 std::vector<std::array<int,3>>fixed(H);
 for(int z=0;z<nf;z++){int e,t;in>>e>>t;if(t<0||t>=H)return 3;fixed[t][e]++;}
 if(!in)return 4;
 struct Escape {int weight;bool before,after;std::vector<std::pair<int,int>>b,a;};
 struct Cut {int need;std::vector<Escape>groups;};
 std::vector<Cut>cuts;int nc=0;
 if(in>>nc)for(int c=0;c<nc;c++){
  Cut cut;int count;in>>cut.need>>count;
  for(int z=0;z<count;z++){
   Escape ex;int nb,na;in>>ex.weight>>nb>>na;ex.before=nb>=0;ex.after=na>=0;
   for(int k=0;k<nb;k++){int j,l;in>>j>>l;ex.b.emplace_back(j,l);}
   for(int k=0;k<na;k++){int j,l;in>>j>>l;ex.a.emplace_back(j,l);}
   cut.groups.push_back(std::move(ex));
  }
  cuts.push_back(std::move(cut));
 }
 std::mt19937_64 rng(std::stoull(argv[4]));int iterations=std::stoi(argv[3]);
 const int caps[3]={2,2,1};
 auto run=[&](const std::vector<int>&old,bool reverse,double noise){
  const auto&deps=reverse?succ:pred;const auto&next=reverse?pred:succ;
  std::vector<int>degree(n),first(n),last(n),times(n,-1);
  using Item=std::pair<double,int>;std::priority_queue<Item,std::vector<Item>,std::greater<Item>>ready;
  using Future=std::pair<int,int>;std::priority_queue<Future,std::vector<Future>,std::greater<Future>>future;
  std::vector<double>pr(n);std::uniform_real_distribution<double>u(-noise,noise);
  for(int j=0;j<n;j++){
   first[j]=reverse?H-1-hi[j]:lo[j];last[j]=reverse?H-1-lo[j]:hi[j];
   pr[j]=(reverse?H-1-old[j]:old[j])+u(rng);
   degree[j]=deps[j].size();
   if(!degree[j])future.emplace(first[j],j);
  }
  int done=0;
  for(int t=0;t<H&&done<n;t++){
   while(!future.empty()&&future.top().first<=t){int j=future.top().second;future.pop();ready.emplace(pr[j],j);}
   auto used=fixed[reverse?H-1-t:t];std::vector<Item>blocked;
   while(!ready.empty()){
    auto item=ready.top();ready.pop();int j=item.second;
    if(t>last[j])return std::vector<int>{};
    if(used[eng[j]]>=caps[eng[j]]){blocked.push_back(item);continue;}
    used[eng[j]]++;times[j]=t;done++;
    for(auto[q,l]:next[j]){
     first[q]=std::max(first[q],t+l);
     if(!--degree[q]){if(first[q]<=t)ready.emplace(pr[q],q);else future.emplace(first[q],q);}
    }
   }
   for(auto item:blocked)ready.push(item);
  }
  if(done!=n)return std::vector<int>{};
  if(reverse)for(int&t:times)t=H-1-t;
  return times;
 };
 std::vector<int>current=hint,best;long long bestcost=1LL<<60;
 const double noise[]={0,.49,1.95,3.95,7.95,15.95,31.95};
 for(int it=0;it<iterations;it++){
  for(int pass=0;pass<2;pass++){
   auto times=run(current,(it+pass)%2,pass?0:noise[it%7]);
   if(times.empty())continue;
   current=times;long long cost=0;
   for(int j=0;j<n;j++)cost+=(eng[j]==0?4:eng[j]==2?2:1)*std::abs(times[j]-hint[j]);
   int deficit=0;
   for(const auto&cut:cuts){
    int escaped=0;
    for(const auto&ex:cut.groups){
     bool before=ex.before,after=ex.after;
     if(before)for(auto[j,l]:ex.b)if(times[j]>l){before=false;break;}
     if(after)for(auto[j,l]:ex.a)if(times[j]<l){after=false;break;}
     if(before||after)escaped+=ex.weight;
    }
    deficit+=std::max(0,cut.need-escaped);
   }
   long long score=cost+1000000LL*deficit;
   if(score<bestcost){
    best=times;bestcost=score;
    std::ofstream out(argv[2]);out<<n<<'\n';for(int j=0;j<n;j++)out<<id[j]<<' '<<best[j]<<'\n';
    std::cout<<"iteration "<<it<<" cost "<<cost<<" energy deficit "<<deficit<<std::endl;
   }
  }
  if(it%16==15)current=best.empty()?hint:best;
 }
 if(best.empty()){std::cout<<"no feasible projected stream"<<std::endl;return 5;}
 for(int j=0;j<n;j++)for(auto[d,l]:pred[j])if(best[j]<best[d]+l)return 6;
 std::cout<<"projected stream only, cost "<<bestcost<<std::endl;
}
