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
 std::vector<double> adjustment(n);
 if(argc>9&&std::string(argv[9])!="-"){std::ifstream af(argv[9]);for(double&v:adjustment)af>>v;if(!af)return 6;}
 std::vector<int> feed(n,999);
 if(argc>10){std::ifstream ff(argv[10]);for(int&v:feed)ff>>v;if(!ff)return 6;}
 const double feed_weight=argc>11?std::stod(argv[11]):0;
 const int caps[5]={12,6,2,2,1};
 auto run=[&](const std::vector<int>&ori,const std::vector<int>&prev,bool reverse,double noise,int defer)->Result{
  const auto&deps=reverse?succ:pred;const auto&next=reverse?pred:succ;
  int high=*std::max_element(ori.begin(),ori.end());
  std::vector<int> target=pins;
  if(reverse)for(int j=0;j<n;j++)if(pins[j]>=0)target[j]=high-pins[j];
  if(reverse&&argc>8&&std::stoi(argv[8])==0)std::fill(target.begin(),target.end(),-1);
  std::vector<double> priority(n);std::uniform_real_distribution<double>u(-noise,noise);
  for(int j=0;j<n;j++)priority[j]=(target[j]>=0)?-1000000.0+target[j]:
   (reverse?high-ori[j]:ori[j])+u(rng)+(engine[j]==0?scalar_bias:0.0)+
   (reverse?-adjustment[j]:adjustment[j]);
  std::vector<int> degree(n),earliest(n),times(n,-1),readycount(ng),firstcycle(ng,-1),issued(ng),mode(ng);
  using Item=std::pair<double,int>;
  std::array<std::priority_queue<Item,std::vector<Item>,std::greater<Item>>,14>ready;
  using Future=std::pair<int,int>;std::priority_queue<Future,std::vector<Future>,std::greater<Future>> future;
  int ready_loads=0;
  auto bucket=[&](int j){return feed[j]<=12?feed[j]:13;};
  auto addready=[&](int j){if(engine[j]==2)ready_loads++;if(gid[j]>=0)readycount[gid[j]]++;ready[bucket(j)].emplace(priority[j],j);};
  for(int j=0;j<n;j++){degree[j]=deps[j].size();if(!degree[j])addready(j);}
  int t=0,done=0;
  while(done<n && t<2500){
   while(!future.empty()&&future.top().first<=t){int j=future.top().second;future.pop();addready(j);}
   int used[5]={},arithmetic=0,scalar=0,fixedv=0,forced=0;
   std::vector<int>fullgroups;std::vector<Item>blocked;
   auto issue=[&](int j){
    times[j]=t;done++;
    if(engine[j]==2)ready_loads--;
    int g=gid[j];if(g>=0){if(firstcycle[g]<0)firstcycle[g]=t;issued[g]++;}
    for(auto[q,l]:next[j]){earliest[q]=std::max(earliest[q],t+l);if(!--degree[q]){if(earliest[q]<=t)addready(q);else future.emplace(earliest[q],q);}}
   };
   while(true){
    int which=-1;Item best_item{1e100,n};
    for(int b=0;b<14;b++)if(!ready[b].empty()){
     auto value=ready[b].top();
     if(!reverse&&ready_loads<4&&b<=12)value.first-=feed_weight*(13-b);
     if(value<best_item){best_item=value;which=b;}
    }
    if(which<0)break;
    auto item=ready[which].top();ready[which].pop();
    auto[pr,j]=best_item;if(times[j]>=0)continue;int g=gid[j];
    if(target[j]>=0&&target[j]>t){blocked.push_back(item);continue;}
    if(target[j]>=0&&target[j]<t)return {9999,{}, {}};
    bool allow_native=true;
    if(g>=0){
     if(pinmode[g]==0)allow_native=false;
     for(int a:groups[g])if(times[a]<0&&target[a]>=0&&target[a]!=t)allow_native=false;
    }
    // Full binary vectors reserve eight lanes first. Their physical engine
    // is chosen after the whole packet, so a late fixed VALU can displace one
    // complete binary vector to eight ALUs. Same-cycle scalar completions
    // can also coalesce before that engine choice.
    if(g>=0&&allow_native&&(firstcycle[g]<0||firstcycle[g]==t)&&readycount[g]==8&&
       arithmetic+8-issued[g]<=60&&(pinmode[g]!=1||fixedv+forced<6)){
     int prior=issued[g];arithmetic+=8-prior;scalar-=prior;
     fullgroups.push_back(g);forced+=pinmode[g]==1;
     for(int a:groups[g])if(times[a]<0)issue(a);
     continue;
    }
    if(g>=0&&pinmode[g]==1){blocked.push_back(item);continue;}
    if(engine[j]==0){
     if(scalar>=12||arithmetic>=60){blocked.push_back(item);continue;}
     if(g>=0&&firstcycle[g]<0&&prev[g]&&readycount[g]<8&&defer>0&&pr+defer>t){blocked.push_back(item);continue;}
     scalar++;arithmetic++;
    }else if(engine[j]==1){
     if(fixedv+forced>=6||arithmetic+8>60){blocked.push_back(item);continue;}
     fixedv++;arithmetic+=8;
    }else{
     if(used[engine[j]]>=caps[engine[j]]){blocked.push_back(item);continue;}
     used[engine[j]]++;
    }
    issue(j);
   }
   std::stable_sort(fullgroups.begin(),fullgroups.end(),[&](int a,int b){
    return (pinmode[a]==1?2:prev[a])>(pinmode[b]==1?2:prev[b]);
   });
   int native=std::min((int)fullgroups.size(),6-fixedv);
   if(scalar+8*((int)fullgroups.size()-native)>12)return {9999,{}, {}};
   for(int k=0;k<(int)fullgroups.size();k++)mode[fullgroups[k]]=k<native;
   for(auto item:blocked)ready[bucket(item.second)].push(item);
   t++;
  }
  if(done!=n)return {9999,{}, {}};
  if(reverse)for(int&x:times)x=t-1-x;
  return {t,std::move(times),std::move(mode)};
 };
 Result best{9999,{},{}},current{*std::max_element(original.begin(),original.end())+1,original,oldmode};
 if(argc>7&&std::string(argv[7])!="-"){
  std::ifstream ss(argv[7]);int sn,sg;ss>>sn>>sg;if(sn!=n||sg!=ng)return 7;
  for(int&t:current.times)ss>>t;for(int&m:current.mode)ss>>m;if(!ss)return 8;
  current.h=*std::max_element(current.times.begin(),current.times.end())+1;
  for(int j=0;j<n;j++)for(auto[d,l]:pred[j])if(current.times[j]<current.times[d]+l)return 9;
  std::vector<std::array<int,5>>use(current.h);
  for(int j=0;j<n;j++){
   if(current.times[j]<0||engine[j]<0||engine[j]>4)return 9;
   int g=gid[j],e=engine[j];
   if(g>=0&&current.mode[g]){
    if(current.mode[g]!=1)return 9;
    for(int a:groups[g])if(current.times[a]!=current.times[j])return 9;
    if(j!=groups[g][0])continue;e=1;
   }
   if(++use[current.times[j]][e]>caps[e])return 9;
  }
  best=current;
 }
 auto quality=[&](const Result&r){long long q=0;
  for(int j=0;j<n;j++)if(engine[j]==0||engine[j]==1)q+=(long long)r.times[j]*(engine[j]==1?8:1);
  return q;
 };
 long long bestq=best.h==9999?0:quality(best);
 const double noises[]={0,.49,1.95,3.95,7.95,15.95,31.95,63.95};const int defers[]={0,1,2,4};
 for(int it=0;it<iterations;it++){
  if(it%16==15&&!best.times.empty())current=best;
  for(int pass=0;pass<4;pass++){
   auto result=run(current.times,current.mode,(it+pass)%2,pass==0?noises[it%8]:0,defers[it%4]);
   if(result.h==9999)continue;
   current=std::move(result);
   long long q=quality(current);
   if(current.h<best.h||(current.h==best.h&&q<bestq)){
    bool shorter=current.h<best.h;best=current;bestq=q;
    if(shorter){
     std::cout<<"iteration "<<it<<" pass "<<pass<<" cycles "<<best.h<<" quality "<<q<<std::endl;
     std::ofstream out(argv[2]);out<<n<<' '<<ng<<'\n';for(int v:best.times)out<<v<<' ';out<<'\n';for(int v:best.mode)out<<v<<' ';out<<'\n';
    }
   }
  }
 }
 if(best.h<9999){
  std::ofstream out(argv[2]);out<<n<<' '<<ng<<'\n';for(int v:best.times)out<<v<<' ';out<<'\n';for(int v:best.mode)out<<v<<' ';out<<'\n';
  std::cout<<"final cycles "<<best.h<<" quality "<<bestq<<std::endl;
 }
}
