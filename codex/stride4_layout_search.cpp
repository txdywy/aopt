#include <algorithm>
#include <iostream>
#include <vector>
using U=__uint128_t;
int pop(U v){return __builtin_popcountll((unsigned long long)v)+__builtin_popcountll((unsigned long long)(v>>64));}
int main(){
 int best=1000;
 for(int x=-20;x<=20;x++)for(int z=-20;z<=20;z++)if(x&&z&&x!=z)
 for(int dy=1;dy<=48;dy++){
  std::vector<int>s={-2*x,-x,dy-2*x,dy-x,-2*z,-z,dy-2*z,dy-z};
  auto a=s;std::sort(a.begin(),a.end());if(std::unique(a.begin(),a.end())!=a.end())continue;
  int lo=a.front(),hi=a.back();if(hi-lo>=60)continue;U b=0;for(int v:s)b|=U(1)<<(v-lo);
  for(int v=-24;v<=24;v++)if(v){
   int l0=std::min({0,-v,-2*v});int h0=std::max({0,-v,-2*v});
   U p=(b<<(-l0))|(b<<(-v-l0))|(b<<(-2*v-l0));if(pop(p)!=24)continue;
   for(int w=v+1;w<=24;w++)if(w){
    int l=std::min({l0,-w,-2*w}),h=std::max({h0,-w,-2*w});
    int span=hi-lo+h-l+1;if(span>=best||span>60)continue;
    U u=(p<<(l0-l))|(b<<(-w-l))|(b<<(-2*w-l));if(pop(u)!=40)continue;
    best=span;std::cout<<"slots "<<best<<" X "<<4*x<<' '<<4*z<<" DY "<<4*dy<<" V "<<4*v<<' '<<4*w<<" LOW "<<4*(lo+l)<<std::endl;
    if(best<=56)return 0;
   }
  }
 }
}
