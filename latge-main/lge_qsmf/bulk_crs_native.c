#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

typedef __uint128_t u128;
static const uint64_t RC[24]={
  0x0000000000000001ULL,0x0000000000008082ULL,0x800000000000808aULL,0x8000000080008000ULL,
  0x000000000000808bULL,0x0000000080000001ULL,0x8000000080008081ULL,0x8000000000008009ULL,
  0x000000000000008aULL,0x0000000000000088ULL,0x0000000080008009ULL,0x000000008000000aULL,
  0x000000008000808bULL,0x800000000000008bULL,0x8000000000008089ULL,0x8000000000008003ULL,
  0x8000000000008002ULL,0x8000000000000080ULL,0x000000000000800aULL,0x800000008000000aULL,
  0x8000000080008081ULL,0x8000000000008080ULL,0x0000000080000001ULL,0x8000000080008008ULL};
static const int ROT[24]={1,3,6,10,15,21,28,36,45,55,2,14,27,41,56,8,25,43,62,18,39,61,20,44};
static const int PIL[24]={10,7,11,17,18,3,5,16,8,21,24,4,15,23,19,13,12,2,20,14,22,9,6,1};
static uint64_t rol(uint64_t x,int b){return (x<<b)|(x>>(64-b));}
static void keccak(uint64_t s[25]){
  uint64_t bc[5],t,tmp;
  for(int round=0;round<24;round++){
    for(int i=0;i<5;i++)bc[i]=s[i]^s[i+5]^s[i+10]^s[i+15]^s[i+20];
    for(int i=0;i<5;i++){t=bc[(i+4)%5]^rol(bc[(i+1)%5],1);for(int j=0;j<25;j+=5)s[j+i]^=t;}
    t=s[1];for(int i=0;i<24;i++){int j=PIL[i];tmp=s[j];s[j]=rol(t,ROT[i]);t=tmp;}
    for(int j=0;j<25;j+=5){for(int i=0;i<5;i++)bc[i]=s[j+i];for(int i=0;i<5;i++)s[j+i]^=(~bc[(i+1)%5])&bc[(i+2)%5];}
    s[0]^=RC[round];
  }
}
typedef struct {uint64_t s[25];size_t pos;} Shake;
static void absorb(Shake *x,const uint8_t *data,size_t count){
  for(size_t i=0;i<count;i++){
    x->s[x->pos>>3]^=(uint64_t)data[i]<<(8*(x->pos&7));
    if(++x->pos==136){keccak(x->s);x->pos=0;}
  }
}
static void absorb_u64(Shake *x,uint64_t value){
  if(x->pos<=128){
    unsigned index=(unsigned)(x->pos>>3),shift=(unsigned)(8*(x->pos&7));
    x->s[index]^=value<<shift;
    if(shift)x->s[index+1]^=value>>(64-shift);
    x->pos+=8;if(x->pos==136){keccak(x->s);x->pos=0;}
    return;
  }
  uint8_t bytes[8];for(int j=0;j<8;j++)bytes[j]=(uint8_t)(value>>(8*j));
  absorb(x,bytes,8);
}
static void finish(Shake *x){
  x->s[x->pos>>3]^=(uint64_t)0x1f<<(8*(x->pos&7));
  x->s[16]^=(uint64_t)0x80<<56;keccak(x->s);x->pos=0;
}
static uint8_t squeeze(Shake *x){
  if(x->pos==136){keccak(x->s);x->pos=0;}
  uint8_t b=(uint8_t)(x->s[x->pos>>3]>>(8*(x->pos&7)));x->pos++;return b;
}
int latge_shake256(const uint8_t *data,size_t length,uint8_t *out,size_t count){
  Shake x={{0},0};absorb(&x,data,length);finish(&x);
  for(size_t i=0;i<count;i++)out[i]=squeeze(&x);return 0;
}
static uint64_t squeeze40(Shake *x){
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  if(x->pos<=131){
    const uint8_t *bytes=(const uint8_t*)x->s+x->pos;uint32_t low;
    memcpy(&low,bytes,4);x->pos+=5;return (uint64_t)low|((uint64_t)bytes[4]<<32);
  }
#endif
  uint64_t candidate=0;
  for(int j=0;j<5;j++)candidate|=(uint64_t)squeeze(x)<<(8*j);
  return candidate;
}
static void sample_entry(Shake *x,uint64_t q,int n,uint64_t *out){
  if(q==UINT64_C(4294967969)){
    for(int i=0;i<n;){
      uint64_t candidate=squeeze40(x);
      if(candidate<UINT64_C(255)*UINT64_C(4294967969)){
        int64_t reduced=(int64_t)(uint32_t)candidate-INT64_C(673)*(int64_t)(candidate>>32);
        out[i++]=(uint64_t)(reduced<0?reduced+INT64_C(4294967969):reduced);
      }
    }
    return;
  }
  int width=1;while((q>>(8*width))!=0)width++;
  uint64_t limit=(((uint64_t)1<<(8*width))/q)*q;
  for(int i=0;i<n;){uint64_t candidate=0;for(int j=0;j<width;j++)candidate|=(uint64_t)squeeze(x)<<(8*j);
    if(candidate<limit)out[i++]=candidate%q;}
}
static void entry(const Shake *prefix,uint64_t row,uint64_t col,uint64_t q,int n,uint64_t *out){
  Shake x=*prefix;absorb_u64(&x,row);absorb_u64(&x,col);finish(&x);
  sample_entry(&x,q,n,out);
}
typedef uint64_t lanes2 __attribute__((vector_size(16)));
static inline lanes2 rol2(lanes2 x,int b){return (x<<b)|(x>>(64-b));}
static void keccak2(lanes2 s[25]){
#if defined(__aarch64__) && defined(__ARM_FEATURE_SHA3)
  const uint64_t *constants=RC;unsigned rounds=24;
  __asm__ volatile(
    "ldp q0, q1, [%[state], #0]\n\t"
    "ldp q2, q3, [%[state], #32]\n\t"
    "ldp q4, q5, [%[state], #64]\n\t"
    "ldp q6, q7, [%[state], #96]\n\t"
    "ldp q8, q9, [%[state], #128]\n\t"
    "ldp q10, q11, [%[state], #160]\n\t"
    "ldp q12, q13, [%[state], #192]\n\t"
    "ldp q14, q15, [%[state], #224]\n\t"
    "ldp q16, q17, [%[state], #256]\n\t"
    "ldp q18, q19, [%[state], #288]\n\t"
    "ldp q20, q21, [%[state], #320]\n\t"
    "ldp q22, q23, [%[state], #352]\n\t"
    "ldr q24, [%[state], #384]\n\t"
    "1:\n\t"
    "eor3 v25.16b, v0.16b, v5.16b, v10.16b\n\t"
    "eor3 v25.16b, v25.16b, v15.16b, v20.16b\n\t"
    "eor3 v26.16b, v1.16b, v6.16b, v11.16b\n\t"
    "eor3 v26.16b, v26.16b, v16.16b, v21.16b\n\t"
    "eor3 v27.16b, v2.16b, v7.16b, v12.16b\n\t"
    "eor3 v27.16b, v27.16b, v17.16b, v22.16b\n\t"
    "eor3 v28.16b, v3.16b, v8.16b, v13.16b\n\t"
    "eor3 v28.16b, v28.16b, v18.16b, v23.16b\n\t"
    "eor3 v29.16b, v4.16b, v9.16b, v14.16b\n\t"
    "eor3 v29.16b, v29.16b, v19.16b, v24.16b\n\t"
    "rax1 v30.2d, v29.2d, v26.2d\n\t"
    "rax1 v31.2d, v25.2d, v27.2d\n\t"
    "rax1 v25.2d, v28.2d, v25.2d\n\t"
    "rax1 v26.2d, v26.2d, v28.2d\n\t"
    "rax1 v27.2d, v27.2d, v29.2d\n\t"
    "mov v28.16b, v1.16b\n\t"
    "eor v0.16b, v0.16b, v30.16b\n\t"
    "xar v1.2d, v6.2d, v31.2d, #20\n\t"
    "xar v6.2d, v9.2d, v25.2d, #44\n\t"
    "xar v9.2d, v22.2d, v26.2d, #3\n\t"
    "xar v22.2d, v14.2d, v25.2d, #25\n\t"
    "xar v14.2d, v20.2d, v30.2d, #46\n\t"
    "xar v20.2d, v2.2d, v26.2d, #2\n\t"
    "xar v2.2d, v12.2d, v26.2d, #21\n\t"
    "xar v12.2d, v13.2d, v27.2d, #39\n\t"
    "xar v13.2d, v19.2d, v25.2d, #56\n\t"
    "xar v19.2d, v23.2d, v27.2d, #8\n\t"
    "xar v23.2d, v15.2d, v30.2d, #23\n\t"
    "xar v15.2d, v4.2d, v25.2d, #37\n\t"
    "xar v4.2d, v24.2d, v25.2d, #50\n\t"
    "xar v24.2d, v21.2d, v31.2d, #62\n\t"
    "xar v21.2d, v8.2d, v27.2d, #9\n\t"
    "xar v8.2d, v16.2d, v31.2d, #19\n\t"
    "xar v16.2d, v5.2d, v30.2d, #28\n\t"
    "xar v5.2d, v3.2d, v27.2d, #36\n\t"
    "xar v3.2d, v18.2d, v27.2d, #43\n\t"
    "xar v18.2d, v17.2d, v26.2d, #49\n\t"
    "xar v17.2d, v11.2d, v31.2d, #54\n\t"
    "xar v11.2d, v7.2d, v26.2d, #58\n\t"
    "xar v7.2d, v10.2d, v30.2d, #61\n\t"
    "xar v10.2d, v28.2d, v31.2d, #63\n\t"
    "mov v25.16b, v0.16b\n\t"
    "mov v26.16b, v1.16b\n\t"
    "bcax v0.16b, v0.16b, v2.16b, v1.16b\n\t"
    "bcax v1.16b, v1.16b, v3.16b, v2.16b\n\t"
    "bcax v2.16b, v2.16b, v4.16b, v3.16b\n\t"
    "bcax v3.16b, v3.16b, v25.16b, v4.16b\n\t"
    "bcax v4.16b, v4.16b, v26.16b, v25.16b\n\t"
    "mov v25.16b, v5.16b\n\t"
    "mov v26.16b, v6.16b\n\t"
    "bcax v5.16b, v5.16b, v7.16b, v6.16b\n\t"
    "bcax v6.16b, v6.16b, v8.16b, v7.16b\n\t"
    "bcax v7.16b, v7.16b, v9.16b, v8.16b\n\t"
    "bcax v8.16b, v8.16b, v25.16b, v9.16b\n\t"
    "bcax v9.16b, v9.16b, v26.16b, v25.16b\n\t"
    "mov v25.16b, v10.16b\n\t"
    "mov v26.16b, v11.16b\n\t"
    "bcax v10.16b, v10.16b, v12.16b, v11.16b\n\t"
    "bcax v11.16b, v11.16b, v13.16b, v12.16b\n\t"
    "bcax v12.16b, v12.16b, v14.16b, v13.16b\n\t"
    "bcax v13.16b, v13.16b, v25.16b, v14.16b\n\t"
    "bcax v14.16b, v14.16b, v26.16b, v25.16b\n\t"
    "mov v25.16b, v15.16b\n\t"
    "mov v26.16b, v16.16b\n\t"
    "bcax v15.16b, v15.16b, v17.16b, v16.16b\n\t"
    "bcax v16.16b, v16.16b, v18.16b, v17.16b\n\t"
    "bcax v17.16b, v17.16b, v19.16b, v18.16b\n\t"
    "bcax v18.16b, v18.16b, v25.16b, v19.16b\n\t"
    "bcax v19.16b, v19.16b, v26.16b, v25.16b\n\t"
    "mov v25.16b, v20.16b\n\t"
    "mov v26.16b, v21.16b\n\t"
    "bcax v20.16b, v20.16b, v22.16b, v21.16b\n\t"
    "bcax v21.16b, v21.16b, v23.16b, v22.16b\n\t"
    "bcax v22.16b, v22.16b, v24.16b, v23.16b\n\t"
    "bcax v23.16b, v23.16b, v25.16b, v24.16b\n\t"
    "bcax v24.16b, v24.16b, v26.16b, v25.16b\n\t"
    "ld1r {v31.2d}, [%[constants]], #8\n\t"
    "eor v0.16b, v0.16b, v31.16b\n\t"
    "subs %w[rounds], %w[rounds], #1\n\t"
    "b.ne 1b\n\t"
    "stp q0, q1, [%[state], #0]\n\t"
    "stp q2, q3, [%[state], #32]\n\t"
    "stp q4, q5, [%[state], #64]\n\t"
    "stp q6, q7, [%[state], #96]\n\t"
    "stp q8, q9, [%[state], #128]\n\t"
    "stp q10, q11, [%[state], #160]\n\t"
    "stp q12, q13, [%[state], #192]\n\t"
    "stp q14, q15, [%[state], #224]\n\t"
    "stp q16, q17, [%[state], #256]\n\t"
    "stp q18, q19, [%[state], #288]\n\t"
    "stp q20, q21, [%[state], #320]\n\t"
    "stp q22, q23, [%[state], #352]\n\t"
    "str q24, [%[state], #384]\n\t"
    : [constants] "+&r"(constants), [rounds] "+&r"(rounds)
    : [state] "r"(s)
    : "memory", "cc", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
#else
  for(int round=0;round<24;round++){
    lanes2 c0=s[0]^s[5]^s[10]^s[15]^s[20];
    lanes2 c1=s[1]^s[6]^s[11]^s[16]^s[21];
    lanes2 c2=s[2]^s[7]^s[12]^s[17]^s[22];
    lanes2 c3=s[3]^s[8]^s[13]^s[18]^s[23];
    lanes2 c4=s[4]^s[9]^s[14]^s[19]^s[24];
    lanes2 d0=c4^rol2(c1,1);
    lanes2 d1=c0^rol2(c2,1);
    lanes2 d2=c1^rol2(c3,1);
    lanes2 d3=c2^rol2(c4,1);
    lanes2 d4=c3^rol2(c0,1);
    s[0]^=d0; s[1]^=d1; s[2]^=d2; s[3]^=d3; s[4]^=d4;
    s[5]^=d0; s[6]^=d1; s[7]^=d2; s[8]^=d3; s[9]^=d4;
    s[10]^=d0; s[11]^=d1; s[12]^=d2; s[13]^=d3; s[14]^=d4;
    s[15]^=d0; s[16]^=d1; s[17]^=d2; s[18]^=d3; s[19]^=d4;
    s[20]^=d0; s[21]^=d1; s[22]^=d2; s[23]^=d3; s[24]^=d4;
    lanes2 t=s[1],tmp;
    tmp=s[10];s[10]=rol2(t,1);t=tmp;
    tmp=s[7];s[7]=rol2(t,3);t=tmp;
    tmp=s[11];s[11]=rol2(t,6);t=tmp;
    tmp=s[17];s[17]=rol2(t,10);t=tmp;
    tmp=s[18];s[18]=rol2(t,15);t=tmp;
    tmp=s[3];s[3]=rol2(t,21);t=tmp;
    tmp=s[5];s[5]=rol2(t,28);t=tmp;
    tmp=s[16];s[16]=rol2(t,36);t=tmp;
    tmp=s[8];s[8]=rol2(t,45);t=tmp;
    tmp=s[21];s[21]=rol2(t,55);t=tmp;
    tmp=s[24];s[24]=rol2(t,2);t=tmp;
    tmp=s[4];s[4]=rol2(t,14);t=tmp;
    tmp=s[15];s[15]=rol2(t,27);t=tmp;
    tmp=s[23];s[23]=rol2(t,41);t=tmp;
    tmp=s[19];s[19]=rol2(t,56);t=tmp;
    tmp=s[13];s[13]=rol2(t,8);t=tmp;
    tmp=s[12];s[12]=rol2(t,25);t=tmp;
    tmp=s[2];s[2]=rol2(t,43);t=tmp;
    tmp=s[20];s[20]=rol2(t,62);t=tmp;
    tmp=s[14];s[14]=rol2(t,18);t=tmp;
    tmp=s[22];s[22]=rol2(t,39);t=tmp;
    tmp=s[9];s[9]=rol2(t,61);t=tmp;
    tmp=s[6];s[6]=rol2(t,20);t=tmp;
    tmp=s[1];s[1]=rol2(t,44);t=tmp;
    { lanes2 b0=s[0],b1=s[1],b2=s[2],b3=s[3],b4=s[4];
      s[0]^=(~b1)&b2; s[1]^=(~b2)&b3; s[2]^=(~b3)&b4; s[3]^=(~b4)&b0; s[4]^=(~b0)&b1; }
    { lanes2 b0=s[5],b1=s[6],b2=s[7],b3=s[8],b4=s[9];
      s[5]^=(~b1)&b2; s[6]^=(~b2)&b3; s[7]^=(~b3)&b4; s[8]^=(~b4)&b0; s[9]^=(~b0)&b1; }
    { lanes2 b0=s[10],b1=s[11],b2=s[12],b3=s[13],b4=s[14];
      s[10]^=(~b1)&b2; s[11]^=(~b2)&b3; s[12]^=(~b3)&b4; s[13]^=(~b4)&b0; s[14]^=(~b0)&b1; }
    { lanes2 b0=s[15],b1=s[16],b2=s[17],b3=s[18],b4=s[19];
      s[15]^=(~b1)&b2; s[16]^=(~b2)&b3; s[17]^=(~b3)&b4; s[18]^=(~b4)&b0; s[19]^=(~b0)&b1; }
    { lanes2 b0=s[20],b1=s[21],b2=s[22],b3=s[23],b4=s[24];
      s[20]^=(~b1)&b2; s[21]^=(~b2)&b3; s[22]^=(~b3)&b4; s[23]^=(~b4)&b0; s[24]^=(~b0)&b1; }
    s[0]^=(lanes2){RC[round],RC[round]};
  }
#endif
}
int latge_counter_xof(const uint8_t *seed,size_t length,uint64_t counter,
                      uint64_t blocks,uint8_t *out){
  if(blocks&&counter>UINT64_MAX-(blocks-1))return -1;
  Shake prefix={{0},0};absorb(&prefix,seed,length);
  for(uint64_t block=0;block<blocks;block+=2){
    int lanes=block+1<blocks?2:1;Shake x[2];lanes2 states[25];
    for(int lane=0;lane<2;lane++){
      x[lane]=prefix;absorb_u64(&x[lane],counter+block+(lane<lanes?lane:0));
      x[lane].s[x[lane].pos>>3]^=(uint64_t)0x1f<<(8*(x[lane].pos&7));
      x[lane].s[16]^=UINT64_C(0x8000000000000000);
    }
    for(int j=0;j<25;j++)states[j]=(lanes2){x[0].s[j],x[1].s[j]};
    for(unsigned offset=0;offset<4096;offset+=136){
      keccak2(states);unsigned take=4096-offset;if(take>136)take=136;
      for(unsigned j=0;j<take/8;j++)for(int lane=0;lane<lanes;lane++){
        uint64_t word=states[j][lane];uint8_t *dst=out+(block+lane)*4096+offset+8*j;
        for(int byte=0;byte<8;byte++)dst[byte]=(uint8_t)(word>>(8*byte));
      }
    }
  }
  return 0;
}

static void entry2(const Shake *prefix,uint64_t row,uint64_t col,uint64_t q,int n,uint64_t out[2][256]){
  Shake x[2];lanes2 states[25];
  if(prefix->pos<=119){
    for(int j=0;j<25;j++)states[j]=(lanes2){prefix->s[j],prefix->s[j]};
    unsigned index=(unsigned)(prefix->pos>>3),shift=(unsigned)(8*(prefix->pos&7));
    lanes2 rows=(lanes2){row,row},cols=(lanes2){col,col+1};
    states[index]^=rows<<shift;states[index+1]^=cols<<shift;
    if(shift){states[index+1]^=rows>>(64-shift);states[index+2]^=cols>>(64-shift);}
    size_t position=prefix->pos+16;
    uint64_t pad=(uint64_t)0x1f<<(8*(position&7));
    states[position>>3]^=(lanes2){pad,pad};
    states[16]^=(lanes2){UINT64_C(0x8000000000000000),UINT64_C(0x8000000000000000)};
  }else{
    for(int lane=0;lane<2;lane++){
      x[lane]=*prefix;absorb_u64(&x[lane],row);absorb_u64(&x[lane],col+lane);
      x[lane].s[x[lane].pos>>3]^=(uint64_t)0x1f<<(8*(x[lane].pos&7));
      x[lane].s[16]^=(uint64_t)0x80<<56;
    }
    for(int j=0;j<25;j++)states[j]=(lanes2){x[0].s[j],x[1].s[j]};
  }
  keccak2(states);
  if(n==16&&q==UINT64_C(4294967969)){
    unsigned used[2]={0,0};
    #pragma clang loop unroll(full)
    for(int i=0;i<16;i++){
      const unsigned word=(5*i)>>3,shift=((5*i)&7)*8;
      lanes2 candidate=states[word]>>shift;
      if(shift>24)candidate|=states[word+1]<<(64-shift);
      candidate&=(lanes2){UINT64_C(0xffffffffff),UINT64_C(0xffffffffff)};
      for(int lane=0;lane<2;lane++){
        uint64_t value=candidate[lane];
        int64_t reduced=(int64_t)(uint32_t)value-INT64_C(673)*(int64_t)(value>>32);
        out[lane][used[lane]]=(uint64_t)(reduced<0?reduced+INT64_C(4294967969):reduced);
        used[lane]+=value<UINT64_C(255)*UINT64_C(4294967969);
      }
    }
    for(int lane=0;lane<2;lane++)if(used[lane]<16){
      for(int j=0;j<25;j++)x[lane].s[j]=states[j][lane];
      x[lane].pos=80;
      sample_entry(&x[lane],q,16-used[lane],out[lane]+used[lane]);
    }
    return;
  }
  for(int j=0;j<25;j++){x[0].s[j]=states[j][0];x[1].s[j]=states[j][1];}
  x[0].pos=x[1].pos=0;
  for(int lane=0;lane<2;lane++)sample_entry(&x[lane],q,n,out[lane]);
}

static int valid(int n,uint64_t q){return n>=4&&n<=256&&!(n&(n-1))&&q>2&&q<((uint64_t)1<<33);}
typedef struct {uint64_t q,reciprocal;} Modulus;
static Modulus modulus(uint64_t q){Modulus m={q,(uint64_t)(((u128)1<<64)/q)};return m;}
static uint64_t mulmod(uint64_t a,uint64_t b,const Modulus *m){
  u128 product=(u128)a*b;
  if(m->q==UINT64_C(4294967969)){
    int64_t folded=(int64_t)(uint32_t)product
      -INT64_C(673)*(int64_t)(uint32_t)(product>>32)
      +INT64_C(452929)*(int64_t)(product>>64)-INT64_C(4294967969);
    uint64_t reduced=(uint64_t)(uint32_t)folded
      -INT64_C(673)*(folded>>32);
    return reduced>=UINT64_C(4294967969)?reduced-UINT64_C(4294967969):reduced;
  }
  uint64_t quotient=(uint64_t)((product*m->reciprocal)>>64);
  uint64_t remainder=(uint64_t)(product-(u128)quotient*m->q);
  while(remainder>=m->q)remainder-=m->q;
  return remainder;
}
static inline uint64_t reduce128(u128 value,uint64_t q){
  if(q!=UINT64_C(4294967969))return (uint64_t)(value%q);
  int64_t folded=(int64_t)(uint32_t)value
    -INT64_C(673)*(int64_t)(uint32_t)(value>>32)
    +INT64_C(452929)*(int64_t)(uint32_t)(value>>64)
    -INT64_C(304821217)*(int64_t)(uint32_t)(value>>96);
  folded=(int64_t)(uint32_t)folded-INT64_C(673)*(folded>>32);
  folded=(int64_t)(uint32_t)folded-INT64_C(673)*(folded>>32);
  return (uint64_t)(folded<0?folded+INT64_C(4294967969):folded);
}
static uint64_t power(uint64_t a,uint64_t b,const Modulus *m){uint64_t r=1;while(b){if(b&1)r=mulmod(r,a,m);a=mulmod(a,a,m);b>>=1;}return r;}
int latge_expand_witness(int kind,uint64_t n,uint64_t alpha1,uint64_t digits,
    uint64_t start,uint64_t count,uint64_t q,const uint64_t *left,uint64_t left_count,
    const uint64_t *right,uint64_t right_count,uint64_t *out){
  if(kind<0||kind>2||!n||!alpha1||!digits||q<=2||q>=((uint64_t)1<<33))return -1;
  uint64_t limit;
  if(kind==2){
    if(right_count%n||alpha1>UINT64_MAX/digits||alpha1*digits>UINT64_MAX/n||
        left_count!=alpha1*digits*n||right_count>UINT64_MAX/digits||right_count*digits>UINT64_MAX/n)return -1;
    limit=right_count*digits*n;
  }else{
    if(left_count>UINT64_MAX/n||(kind==1&&right_count!=n))return -1;
    limit=left_count*n;
  }
  if(start>limit||count>limit-start)return -1;
  Modulus mod=modulus(q);uint64_t written=0;
  while(written<count){
    uint64_t position=start+written,row=position/n,column=position%n;
    uint64_t take=n-column;if(take>count-written)take=count-written;
    uint64_t factor;const uint64_t *values=NULL;
    if(kind==2){
      uint64_t polynomial=row/n,key=polynomial/digits,digit=polynomial%digits;
      factor=left[((key%alpha1)*digits+digit)*n+row%n];
      values=right+key*n+column;
    }else{factor=left[row];if(kind==1)values=right+column;}
    uint64_t *target=out+written;
    if(kind==0){for(uint64_t j=0;j<take;j++)target[j]=factor;}
    else if(factor==0)memset(target,0,take*sizeof(uint64_t));
    else if(factor==1)memcpy(target,values,take*sizeof(uint64_t));
    else if(factor==q-1){for(uint64_t j=0;j<take;j++)target[j]=values[j]?q-values[j]:0;}
    else{for(uint64_t j=0;j<take;j++)target[j]=mulmod(factor,values[j],&mod);}
    written+=take;
  }
  return 0;
}
int latge_collapse_products(const uint64_t *values,const uint64_t *weights,
    uint64_t polys,uint64_t digits,int n,uint64_t q,uint64_t *out){
  if(!valid(n,q)||!digits||digits>64)return -1;
  Modulus mod=modulus(q);
  for(uint64_t p=0;p<polys;p++){
    uint64_t *result=out+p*n;memset(result,0,n*sizeof(uint64_t));
    for(uint64_t digit=0;digit<digits;digit++){
      int64_t diagonal[256]={0};
      const uint64_t *matrix=values+(p*digits+digit)*n*n;
      for(int i=0;i<n;i++){
        const uint64_t *row=matrix+i*n;
        for(int j=0;j<n-i;j++)diagonal[i+j]+=(int64_t)row[j];
        for(int j=n-i;j<n;j++)diagonal[i+j-n]-=(int64_t)row[j];
      }
      for(int j=0;j<n;j++){
        int64_t reduced=diagonal[j]%(int64_t)q;if(reduced<0)reduced+=(int64_t)q;
        uint64_t value=result[j]+mulmod((uint64_t)reduced,weights[digit],&mod);
        result[j]=value>=q?value-q:value;
      }
    }
  }
  return 0;
}
int latge_check_witness_polynomial(int kind,const uint64_t *left,const uint64_t *right,
    const uint64_t *output,const uint64_t *indicator,uint64_t count,uint64_t period,uint64_t q){
  if(kind<0||kind>2||q<=2||q>=((uint64_t)1<<33)||(kind==0&&(!period||period>count)))return -1;
  Modulus mod=modulus(q);uint64_t position=0;
  for(uint64_t i=0;i<count;i++){
    uint64_t x=left[i],expected;
    if(kind==0){
      uint64_t y=right[position];if(++position==period)position=0;
      expected=x==0?0:x==1?y:x==q-1?(y?q-y:0):mulmod(x,y,&mod);
      if(expected!=output[i])return 1;
    }else if(kind==1){
      if(x!=0&&x!=1&&x!=q-1&&mulmod(mulmod(x,x,&mod),x,&mod)!=x)return 1;
    }else if(mulmod(x,x,&mod)!=(indicator[i]?x:0))return 1;
  }
  return 0;
}
typedef struct {int n;Modulus mod;
  uint64_t twist[256],untwist[256],forward[256],backward[256];
  uint64_t twist_shoup[256],untwist_shoup[256],forward_shoup[256],backward_shoup[256];
  uint64_t negacyclic16[16],negacyclic16_shoup[16];
  uint64_t inverse16[16],inverse16_shoup[16];
} Plan;
static inline uint64_t rootmul(uint64_t a,uint64_t root,uint64_t shoup,uint64_t q){
  uint64_t quotient=(uint64_t)(((u128)a*shoup)>>64);
  uint64_t remainder=a*root-quotient*q;
  return remainder>=q?remainder-q:remainder;
}
static void plan_init(Plan *p,int n,uint64_t q,uint64_t psi){
  p->n=n;p->mod=modulus(q);const Modulus *m=&p->mod;
  uint64_t invn=power(n,q-2,m),invpsi=power(psi,q-2,m);
  p->twist[0]=p->untwist[0]=1;
  for(int i=1;i<n;i++){p->twist[i]=mulmod(p->twist[i-1],psi,m);p->untwist[i]=mulmod(p->untwist[i-1],invpsi,m);}
  for(int i=0;i<n;i++)p->untwist[i]=mulmod(p->untwist[i],invn,m);
  for(int size=2;size<=n;size<<=1){
    int half=size/2;
    uint64_t f=power(mulmod(psi,psi,m),n/size,m);
    uint64_t b=power(mulmod(invpsi,invpsi,m),n/size,m);
    p->forward[half]=p->backward[half]=1;
    for(int j=1;j<half;j++){
      p->forward[half+j]=mulmod(p->forward[half+j-1],f,m);
      p->backward[half+j]=mulmod(p->backward[half+j-1],b,m);
    }
  }
  for(int i=0;i<n;i++){
    p->twist_shoup[i]=(uint64_t)(((u128)p->twist[i]<<64)/q);
    p->untwist_shoup[i]=(uint64_t)(((u128)p->untwist[i]<<64)/q);
    if(i){
      p->forward_shoup[i]=(uint64_t)(((u128)p->forward[i]<<64)/q);
      p->backward_shoup[i]=(uint64_t)(((u128)p->backward[i]<<64)/q);
    }
  }
  if(n==16)for(unsigned i=1;i<16;i++){
    unsigned reversed=((i&1)<<3)|((i&2)<<1)|((i&4)>>1)|((i&8)>>3);
    p->negacyclic16[i]=p->twist[reversed];
    p->negacyclic16_shoup[i]=p->twist_shoup[reversed];
    p->inverse16[i]=power(invpsi,reversed,m);
    p->inverse16_shoup[i]=(uint64_t)(((u128)p->inverse16[i]<<64)/q);
  }
}
static inline void transform16(const Plan *p,uint64_t *a,int inverse){
  const Modulus *m=&p->mod;uint64_t q=m->q;
  if(!inverse){
    unsigned k=1;uint64_t twice=2*q;
    #define NEGACYCLIC16(H) do{ \
      for(int start=0;start<16;start+=2*(H)){ \
        uint64_t root=p->negacyclic16[k],shoup=p->negacyclic16_shoup[k];k++; \
        for(int j=start;j<start+(H);j++){ \
          uint64_t u=a[j],v=a[j+(H)]; \
          if(u>=twice)u-=twice; \
          uint64_t quotient=(uint64_t)(((u128)v*shoup)>>64); \
          v=v*root-quotient*q; \
          a[j]=u+v;a[j+(H)]=u+twice-v; \
        } \
      } \
    }while(0)
    NEGACYCLIC16(8);NEGACYCLIC16(4);NEGACYCLIC16(2);NEGACYCLIC16(1);
    #undef NEGACYCLIC16
    #pragma clang loop unroll(full)
    for(int i=0;i<16;i++){
      uint64_t value=a[i];if(value>=twice)value-=twice;
      a[i]=value>=q?value-q:value;
    }
    #define REORDER16(I,J) do{uint64_t value=a[I];a[I]=a[J];a[J]=value;}while(0)
    REORDER16(1,8);REORDER16(2,4);REORDER16(3,12);REORDER16(5,10);REORDER16(7,14);REORDER16(11,13);
    #undef REORDER16
    return;
  }
  #define SWAP16(I,J) do{uint64_t t=a[I];a[I]=a[J];a[J]=t;}while(0)
  SWAP16(1,8);SWAP16(2,4);SWAP16(3,12);SWAP16(5,10);SWAP16(7,14);SWAP16(11,13);
  #undef SWAP16
  uint64_t twice=2*q;
  #define INVERSE16(H) do{ \
    unsigned k=8/(H); \
    for(int start=0;start<16;start+=2*(H)){ \
      uint64_t root=p->inverse16[k],shoup=p->inverse16_shoup[k];k++; \
      for(int j=start;j<start+(H);j++){ \
        uint64_t u=a[j],v=a[j+(H)],plus=u+v,difference=u+twice-v; \
        uint64_t quotient=(uint64_t)(((u128)difference*shoup)>>64); \
        a[j]=plus>=twice?plus-twice:plus; \
        a[j+(H)]=difference*root-quotient*q; \
      } \
    } \
  }while(0)
  INVERSE16(1);INVERSE16(2);INVERSE16(4);INVERSE16(8);
  #undef INVERSE16
  #pragma clang loop unroll(full)
  for(int i=0;i<16;i++)a[i]=rootmul(a[i],p->untwist[0],p->untwist_shoup[0],q);
}
static void transform(const Plan *p,uint64_t *a,int inverse){
  if(p->n==16){transform16(p,a,inverse);return;}
  int n=p->n;const Modulus *m=&p->mod;uint64_t q=m->q;
  if(!inverse)for(int i=1;i<n;i++)a[i]=rootmul(a[i],p->twist[i],p->twist_shoup[i],q);
  for(int i=1,j=0;i<n;i++){
    int bit=n>>1;for(;j&bit;bit>>=1)j^=bit;j^=bit;
    if(i<j){uint64_t t=a[i];a[i]=a[j];a[j]=t;}
  }
  for(int size=2;size<=n;size<<=1){
    const uint64_t *twiddle=(inverse?p->backward:p->forward)+size/2;
    const uint64_t *shoup=(inverse?p->backward_shoup:p->forward_shoup)+size/2;
    for(int start=0;start<n;start+=size){
      uint64_t u0=a[start],v0=a[start+size/2];
      uint64_t plus0=u0+v0,minus0=u0+q-v0;
      a[start]=plus0>=q?plus0-q:plus0;
      a[start+size/2]=minus0>=q?minus0-q:minus0;
      for(int j=1;j<size/2;j++){
        uint64_t u=a[start+j],v=rootmul(a[start+j+size/2],twiddle[j],shoup[j],q);
        uint64_t plus=u+v,minus=u+q-v;
        a[start+j]=plus>=q?plus-q:plus;
        a[start+j+size/2]=minus>=q?minus-q:minus;
      }
    }
  }
  if(inverse)for(int i=0;i<n;i++)a[i]=rootmul(a[i],p->untwist[i],p->untwist_shoup[i],q);
}
int latge_transform(const uint64_t *input,uint64_t count,int n,uint64_t q,uint64_t psi,int inverse,uint64_t *out){
  if(!valid(n,q))return -1;Plan plan;plan_init(&plan,n,q,psi);
  for(uint64_t i=0;i<count;i++){memcpy(out+i*n,input+i*n,n*8);transform(&plan,out+i*n,inverse);}return 0;
}
int latge_entries(const uint8_t *prefix,size_t length,uint64_t first_row,uint64_t rows,uint64_t first_col,uint64_t cols,int n,uint64_t q,uint64_t *out){
  if(!valid(n,q))return -1;Shake x={{0},0};absorb(&x,prefix,length);
  uint64_t poly[2][256];
  for(uint64_t row=0;row<rows;row++){
    uint64_t col=0;
    for(;col+1<cols;col+=2){
      entry2(&x,first_row+row,first_col+col,q,n,poly);
      for(int lane=0;lane<2;lane++)memcpy(out+(row*cols+col+lane)*n,poly[lane],n*8);
    }
    if(col<cols)entry(&x,first_row+row,first_col+col,q,n,out+(row*cols+col)*n);
  }
  return 0;
}
#if defined(__aarch64__)
#include <arm_neon.h>
#define CRS_TILE_COLUMNS 32
#define MAKE_TILE_ACCUMULATE(BATCH) \
static void tile_accumulate##BATCH(const uint32_t matrix[CRS_TILE_COLUMNS][16], \
    const uint32_t *packed,uint64_t columns,uint64_t start,uint64_t count, \
    uint64_t low[5][16],uint64_t high[5][16]){ \
  const uint64x2_t one=vdupq_n_u64(1); \
  for(int j=0;j<16;j+=4){ \
    uint64x2_t sums[BATCH][2],carry[BATCH][2]; \
    for(int b=0;b<BATCH;b++)for(int h=0;h<2;h++){ \
      sums[b][h]=vld1q_u64(low[b]+j+2*h);carry[b][h]=vld1q_u64(high[b]+j+2*h); \
    } \
    uint64_t col=0; \
    for(;col+1<count;col+=2){ \
      uint32x2_t a0=vld1_u32(matrix[col]+j),a1=vld1_u32(matrix[col]+j+2); \
      uint32x2_t c0=vld1_u32(matrix[col+1]+j),c1=vld1_u32(matrix[col+1]+j+2); \
      for(int b=0;b<BATCH;b++){ \
        const uint32_t *v=packed+(b*columns+start+col)*16+j; \
        uint64x2_t p0=vmull_u32(a0,vld1_u32(v)),p1=vmull_u32(a1,vld1_u32(v+2)); \
        uint64x2_t t0=vaddq_u64(p0,vmull_u32(c0,vld1_u32(v+16))); \
        uint64x2_t t1=vaddq_u64(p1,vmull_u32(c1,vld1_u32(v+18))); \
        uint64x2_t k0=vandq_u64(vcltq_u64(t0,p0),one),k1=vandq_u64(vcltq_u64(t1,p1),one); \
        uint64x2_t s0=vaddq_u64(sums[b][0],t0),s1=vaddq_u64(sums[b][1],t1); \
        k0=vaddq_u64(k0,vandq_u64(vcltq_u64(s0,sums[b][0]),one)); \
        k1=vaddq_u64(k1,vandq_u64(vcltq_u64(s1,sums[b][1]),one)); \
        sums[b][0]=s0;sums[b][1]=s1; \
        carry[b][0]=vaddq_u64(carry[b][0],k0);carry[b][1]=vaddq_u64(carry[b][1],k1); \
      } \
    } \
    if(col<count)for(int b=0;b<BATCH;b++)for(int h=0;h<2;h++){ \
      const uint32_t *v=packed+(b*columns+start+col)*16+j+2*h; \
      uint64x2_t product=vmull_u32(vld1_u32(matrix[col]+j+2*h),vld1_u32(v)); \
      uint64x2_t total=vaddq_u64(sums[b][h],product); \
      carry[b][h]=vaddq_u64(carry[b][h],vandq_u64(vcltq_u64(total,sums[b][h]),one));sums[b][h]=total; \
    } \
    for(int b=0;b<BATCH;b++)for(int h=0;h<2;h++){ \
      vst1q_u64(low[b]+j+2*h,sums[b][h]);vst1q_u64(high[b]+j+2*h,carry[b][h]); \
    } \
  } \
}
MAKE_TILE_ACCUMULATE(4)
MAKE_TILE_ACCUMULATE(5)
#undef MAKE_TILE_ACCUMULATE
static int crs_dot16_neon(const Shake *prefix,const Plan *plan,uint64_t first_row,uint64_t rows,uint64_t first_col,uint64_t cols,const uint64_t *const *vectors,uint64_t batch,const uint64_t *const *identity,uint64_t *const *out){
  if(!cols){
    for(uint64_t b=0;b<batch;b++)for(uint64_t i=0;i<rows*16;i++)out[b][i]=identity?identity[b][i]:0;
    return 0;
  }
  uint32_t *packed=malloc(batch*cols*66);if(!packed)return -2;
  uint16_t *input_high=(uint16_t*)(packed+batch*cols*16);uint16_t input_any=0;
  for(uint64_t b=0;b<batch;b++)for(uint64_t col=0;col<cols;col++){
    uint16_t bits=0;
    for(int j=0;j<16;j++){
      uint64_t value=vectors[b][col*16+j];packed[(b*cols+col)*16+j]=(uint32_t)value;
      bits|=(uint16_t)((value>>32)<<j);
    }
    input_high[b*cols+col]=bits;input_any|=bits;
  }
  uint64_t q=plan->mod.q,poly[2][256];
  uint32_t matrix[CRS_TILE_COLUMNS][16];uint16_t matrix_high[CRS_TILE_COLUMNS];
  for(uint64_t row=0;row<rows;row++){
    uint64_t low[5][16]={{0}},high[5][16]={{0}};
    for(uint64_t start=0;start<cols;start+=CRS_TILE_COLUMNS){
      uint64_t count=cols-start;if(count>CRS_TILE_COLUMNS)count=CRS_TILE_COLUMNS;
      uint16_t matrix_any=0;
      for(uint64_t col=0;col<count;col+=2){
        int lanes=col+1<count?2:1;
        if(lanes==2)entry2(prefix,first_row+row,first_col+start+col,q,16,poly);
        else entry(prefix,first_row+row,first_col+start+col,q,16,poly[0]);
        for(int lane=0;lane<lanes;lane++){
          transform16(plan,poly[lane],0);uint16_t bits=0;
          for(int j=0;j<16;j++){
            matrix[col+lane][j]=(uint32_t)poly[lane][j];bits|=(uint16_t)((poly[lane][j]>>32)<<j);
          }
          matrix_high[col+lane]=bits;matrix_any|=bits;
        }
      }
      if(batch==4)tile_accumulate4(matrix,packed,cols,start,count,low,high);
      else tile_accumulate5(matrix,packed,cols,start,count,low,high);
      if(matrix_any||input_any)for(uint64_t col=0;col<count;col++)for(uint64_t b=0;b<batch;b++){
        uint16_t ah=matrix_high[col],vh=input_high[b*cols+start+col],bits=ah|vh;
        while(bits){
          unsigned j=(unsigned)__builtin_ctz((unsigned)bits);bits&=(uint16_t)(bits-1);
          uint64_t a=matrix[col][j],v=packed[(b*cols+start+col)*16+j];
          uint64_t a_high=(ah>>j)&1,v_high=(vh>>j)&1;
          u128 correction=((u128)(a_high*v+v_high*a)<<32)+((u128)(a_high*v_high)<<64);
          u128 total=((u128)high[b][j]<<64)|low[b][j];total+=correction;
          low[b][j]=(uint64_t)total;high[b][j]=(uint64_t)(total>>64);
        }
      }
    }
    for(uint64_t b=0;b<batch;b++)for(int j=0;j<16;j++){
      uint64_t index=row*16+j;
      out[b][index]=reduce128((((u128)high[b][j]<<64)|low[b][j])+(identity?identity[b][index]:0),q);
    }
  }
  free(packed);return 0;
}
#undef CRS_TILE_COLUMNS

#endif

static int crs_dot_split(const Shake *prefix,const Plan *plan,uint64_t first_row,uint64_t rows,uint64_t first_col,uint64_t cols,const uint64_t *const *vectors,uint64_t batch,const uint64_t *const *identity,uint64_t *const *out){
  int n=plan->n;uint64_t q=plan->mod.q;
  if(!valid(n,q)||rows>4096||cols>65536||batch<1||batch>16)return -1;
  #if defined(__aarch64__)
  if(n==16&&(batch==4||batch==5)){
    return crs_dot16_neon(prefix,plan,first_row,rows,first_col,cols,vectors,batch,identity,out);
  }
#endif
  size_t count=batch*rows*n;u128 *acc=calloc(count,sizeof(u128));if(!acc)return -2;
  uint64_t poly[2][256];
  for(uint64_t row=0;row<rows;row++)for(uint64_t col=0;col<cols;col+=2){
    int lanes=col+1<cols?2:1;
    if(lanes==2)entry2(prefix,first_row+row,first_col+col,q,n,poly);
    else entry(prefix,first_row+row,first_col+col,q,n,poly[0]);
    for(int lane=0;lane<lanes;lane++){
      transform(plan,poly[lane],0);
      for(uint64_t b=0;b<batch;b++)for(int j=0;j<n;j++)
        acc[(b*rows+row)*n+j]+=(u128)poly[lane][j]*vectors[b][(col+lane)*n+j];
    }
  }
  for(uint64_t b=0;b<batch;b++)for(uint64_t i=0;i<rows*n;i++)
    out[b][i]=reduce128(acc[b*rows*n+i]+(identity?identity[b][i]:0),q);
  free(acc);return 0;
}
static int crs_dot_prepared(const Shake *prefix,const Plan *plan,uint64_t first_row,uint64_t rows,uint64_t first_col,uint64_t cols,const uint64_t *vectors,uint64_t batch,const uint64_t *identity,uint64_t *out){
  if(batch<1||batch>16)return -1;
  const uint64_t *v[16],*id[16];uint64_t *target[16];
  for(uint64_t b=0;b<batch;b++){
    v[b]=vectors+b*cols*plan->n;id[b]=identity?identity+b*rows*plan->n:NULL;
    target[b]=out+b*rows*plan->n;
  }
  return crs_dot_split(prefix,plan,first_row,rows,first_col,cols,v,batch,identity?id:NULL,target);
}
static int crs_dot(const uint8_t *prefix,size_t length,uint64_t first_row,uint64_t rows,uint64_t first_col,uint64_t cols,int n,uint64_t q,uint64_t psi,const uint64_t *vectors,uint64_t batch,const uint64_t *identity,uint64_t *out){
  if(!valid(n,q))return -1;
  Shake x={{0},0};absorb(&x,prefix,length);Plan plan;plan_init(&plan,n,q,psi);
  return crs_dot_prepared(&x,&plan,first_row,rows,first_col,cols,vectors,batch,identity,out);
}
int latge_crs_dot(const uint8_t *prefix,size_t length,uint64_t first_row,uint64_t rows,uint64_t first_col,uint64_t cols,int n,uint64_t q,uint64_t psi,const uint64_t *vectors,uint64_t batch,uint64_t *out){
  return crs_dot(prefix,length,first_row,rows,first_col,cols,n,q,psi,vectors,batch,NULL,out);
}
int latge_crs_dot_identity(const uint8_t *prefix,size_t length,uint64_t first_row,uint64_t rows,uint64_t first_col,uint64_t cols,int n,uint64_t q,uint64_t psi,const uint64_t *vectors,uint64_t batch,const uint64_t *identity,uint64_t *out){
  return crs_dot(prefix,length,first_row,rows,first_col,cols,n,q,psi,vectors,batch,identity,out);
}

typedef struct {Plan plan;Shake prefix[2];} CRSContext;
void *latge_context_create(int n,uint64_t q,uint64_t psi,const uint8_t *a1,size_t len1,const uint8_t *a2,size_t len2){
  if(!valid(n,q))return NULL;
  CRSContext *context=calloc(1,sizeof(CRSContext));if(!context)return NULL;
  plan_init(&context->plan,n,q,psi);
  absorb(&context->prefix[0],a1,len1);absorb(&context->prefix[1],a2,len2);
  return context;
}
void latge_context_destroy(void *context){free(context);}
int latge_context_prepare_matrix(const CRSContext *context,int kind,uint64_t first_row,
    uint64_t rows,uint64_t first_col,uint64_t cols,uint32_t *low,uint16_t *high){
  if(!context||kind<0||kind>1||context->plan.n!=16||rows>4096||cols>65536)return -1;
  uint64_t poly[2][256];const Plan *plan=&context->plan;
  for(uint64_t row=0;row<rows;row++)for(uint64_t col=0;col<cols;col+=2){
    int lanes=col+1<cols?2:1;
    if(lanes==2)entry2(&context->prefix[kind],first_row+row,first_col+col,plan->mod.q,16,poly);
    else entry(&context->prefix[kind],first_row+row,first_col+col,plan->mod.q,16,poly[0]);
    for(int lane=0;lane<lanes;lane++){
      transform(plan,poly[lane],0);uint64_t index=row*cols+col+lane;uint16_t bits=0;
      for(int j=0;j<16;j++){
        low[index*16+j]=(uint32_t)poly[lane][j];
        bits|=(uint16_t)((poly[lane][j]>>32)<<j);
      }
      high[index]=bits;
    }
  }
  return 0;
}
static int cached_a2_scalar(const CRSContext *context,const uint32_t *low,
    const uint16_t *high,uint64_t rows,uint64_t cols,const uint64_t *const *vectors,
    uint64_t batch,const uint64_t *const *identity,uint64_t *const *out){
  if(!context||context->plan.n!=16||rows>4096||cols>65536||batch<1||batch>16)return -1;
  uint64_t q=context->plan.mod.q;
  for(uint64_t row=0;row<rows;row++){
    u128 acc[16][16]={{0}};
    for(uint64_t col=0;col<cols;col++){
      uint64_t index=row*cols+col;uint16_t bits=high[index];
      for(uint64_t b=0;b<batch;b++)for(int j=0;j<16;j++){
        uint64_t slot=(uint64_t)low[index*16+j]|((uint64_t)((bits>>j)&1)<<32);
        acc[b][j]+=(u128)slot*vectors[b][col*16+j];
      }
    }
    for(uint64_t b=0;b<batch;b++)for(int j=0;j<16;j++){
      uint64_t index=row*16+j;
      out[b][index]=reduce128(acc[b][j]+(identity?identity[b][index]:0),q);
    }
  }
  return 0;
}

static inline int unpack_row(const uint8_t *,int,uint64_t,uint64_t,uint64_t *);

#if defined(__aarch64__)
typedef struct {
  const uint8_t *raw;
  uint64_t stride,rows;
  int bits;
  uint64_t bound;
  uint32_t *packed;
  uint16_t *high,any;
} ResponseProjection;

#define MAKE_SHORT_DOT(BATCH) \
static void short_dot##BATCH(const uint32_t *matrix,const uint32_t *packed, \
    uint64_t cols,u128 result[5][16]){ \
  for(int j=0;j<16;j+=4){ \
    uint64x2_t low[BATCH][2],high[BATCH][2]; \
    for(int b=0;b<BATCH;b++)for(int h=0;h<2;h++) \
      low[b][h]=high[b][h]=vdupq_n_u64(0); \
    for(uint64_t col=0;col<cols;col++){ \
      uint32x4_t a=vld1q_u32(matrix+col*16+j); \
      for(int b=0;b<BATCH;b++){ \
        uint32x4_t lo=vld1q_u32(packed+(b*cols+col)*16+j); \
        uint32x4_t hi=vld1q_u32(packed+((BATCH+b)*cols+col)*16+j); \
        low[b][0]=vmlal_u32(low[b][0],vget_low_u32(a),vget_low_u32(lo)); \
        low[b][1]=vmlal_high_u32(low[b][1],a,lo); \
        high[b][0]=vmlal_u32(high[b][0],vget_low_u32(a),vget_low_u32(hi)); \
        high[b][1]=vmlal_high_u32(high[b][1],a,hi); \
      } \
    } \
    for(int b=0;b<BATCH;b++)for(int h=0;h<2;h++){ \
      uint64_t lo[2],hi[2];vst1q_u64(lo,low[b][h]);vst1q_u64(hi,high[b][h]); \
      for(int k=0;k<2;k++)result[b][j+2*h+k]=(u128)lo[k]+((u128)hi[k]<<16); \
    } \
  } \
}
MAKE_SHORT_DOT(4)
MAKE_SHORT_DOT(5)
#undef MAKE_SHORT_DOT

static uint16_t pack_short_vectors(const uint64_t *const *vectors,uint64_t batch,
    uint64_t cols,uint32_t *packed,uint16_t *input_high){
  uint16_t input_any=0;
  for(uint64_t b=0;b<batch;b++)for(uint64_t col=0;col<cols;col++){
    uint16_t bits=0;
    for(int j=0;j<16;j++){
      uint64_t value=vectors[b][col*16+j];
      packed[(b*cols+col)*16+j]=value&65535;
      packed[((batch+b)*cols+col)*16+j]=(value>>16)&65535;
      bits|=(uint16_t)((value>>32)<<j);
    }
    input_high[b*cols+col]=bits;input_any|=bits;
  }
  return input_any;
}
static inline __attribute__((always_inline)) int cached_short_impl(const CRSContext *context,const uint32_t *matrix,
    const uint16_t *matrix_high,uint64_t rows,uint64_t cols,
    const uint32_t *packed,const uint16_t *input_high,uint16_t input_any,uint64_t batch,
    const uint64_t *const *identity,uint64_t *const *out,ResponseProjection *response,int decode){
  uint64_t q=context->plan.mod.q;
  for(uint64_t row=0;row<rows;row++){
    u128 acc[5][16];
    if(batch==4)short_dot4(matrix+row*cols*16,packed,cols,acc);
    else short_dot5(matrix+row*cols*16,packed,cols,acc);
    for(uint64_t col=0;col<cols;col++)if(matrix_high[row*cols+col]||input_any)
      for(uint64_t b=0;b<batch;b++){
        uint16_t ah=matrix_high[row*cols+col],vh=input_high[b*cols+col],bits=ah|vh;
        while(bits){
          unsigned j=(unsigned)__builtin_ctz((unsigned)bits);bits&=(uint16_t)(bits-1);
          uint64_t a=matrix[(row*cols+col)*16+j];
          uint64_t v=packed[(b*cols+col)*16+j]|(packed[((batch+b)*cols+col)*16+j]<<16);
          acc[b][j]+=((u128)(((ah>>j)&1)*v+((vh>>j)&1)*a)<<32)
                    +((u128)(((ah&vh)>>j)&1)<<64);
        }
      }
    for(uint64_t b=0;b<batch;b++){
      uint64_t slots[16];
      if(decode){
        const uint8_t *raw=response->raw+b*response->stride+row*2*response->bits;
        if(unpack_row(raw,response->bits,q,response->bound,slots))return -1;
        transform16(&context->plan,slots,0);
        uint16_t mask=0;
        for(int j=0;j<16;j++){
          response->packed[(b*response->rows+row)*16+j]=(uint32_t)slots[j];
          mask|=(uint16_t)((slots[j]>>32)<<j);
        }
        response->high[b*response->rows+row]=mask;response->any|=mask;
      }
      for(int j=0;j<16;j++){
        uint64_t index=row*16+j;
        uint64_t value=decode?slots[j]:(identity?identity[b][index]:0);
        out[b][index]=reduce128(acc[b][j]+value,q);
      }
    }
  }
  return 0;
}
static int cached_short_packed(const CRSContext *context,const uint32_t *matrix,
    const uint16_t *matrix_high,uint64_t rows,uint64_t cols,
    const uint32_t *packed,const uint16_t *input_high,uint16_t input_any,uint64_t batch,
    const uint64_t *const *identity,uint64_t *const *out,ResponseProjection *response){
  if(response){
    if(batch==4)return cached_short_impl(context,matrix,matrix_high,rows,cols,
        packed,input_high,input_any,4,identity,out,response,1);
    return cached_short_impl(context,matrix,matrix_high,rows,cols,
        packed,input_high,input_any,5,identity,out,response,1);
  }
  if(batch==4)return cached_short_impl(context,matrix,matrix_high,rows,cols,
      packed,input_high,input_any,4,identity,out,NULL,0);
  return cached_short_impl(context,matrix,matrix_high,rows,cols,
      packed,input_high,input_any,5,identity,out,NULL,0);
}
static int cached_short_dot(const CRSContext *context,const uint32_t *matrix,
    const uint16_t *matrix_high,uint64_t rows,uint64_t cols,
    const uint64_t *const *vectors,uint64_t batch,
    const uint64_t *const *identity,uint64_t *const *out){
  uint32_t *packed=malloc(batch*cols*130);if(!packed)return -2;
  uint16_t *input_high=(uint16_t*)(packed+batch*cols*32);
  uint16_t input_any=pack_short_vectors(vectors,batch,cols,packed,input_high);
  int code=cached_short_packed(context,matrix,matrix_high,rows,cols,
                             packed,input_high,input_any,batch,identity,out,NULL);
  free(packed);return code;
}

static uint16_t pack_long_vectors(const uint64_t *const *vectors,uint64_t batch,
    uint64_t cols,uint32_t *packed,uint16_t *input_high){
  uint16_t input_any=0;
  for(uint64_t b=0;b<batch;b++)for(uint64_t col=0;col<cols;col++){
    uint16_t bits=0;
    for(int j=0;j<16;j++){
      uint64_t value=vectors[b][col*16+j];packed[(b*cols+col)*16+j]=(uint32_t)value;
      bits|=(uint16_t)((value>>32)<<j);
    }
    input_high[b*cols+col]=bits;input_any|=bits;
  }
  return input_any;
}
static int cached_long_packed(const CRSContext *context,const uint32_t *matrix,
    const uint16_t *matrix_high,uint64_t rows,uint64_t cols,
    const uint32_t *packed,const uint16_t *input_high,uint16_t input_any,uint64_t batch,
    const uint64_t *const *identity,uint64_t *const *out){
  uint64_t q=context->plan.mod.q;
  for(uint64_t row=0;row<rows;row++){
    uint64_t low[5][16]={{0}},high[5][16]={{0}};
    for(uint64_t start=0;start<cols;start+=32){
      uint64_t count=cols-start;if(count>32)count=32;
      const uint32_t (*tile)[16]=(const uint32_t (*)[16])(matrix+(row*cols+start)*16);
      if(batch==4)tile_accumulate4(tile,packed,cols,start,count,low,high);
      else tile_accumulate5(tile,packed,cols,start,count,low,high);
      uint16_t matrix_any=0;
      for(uint64_t col=0;col<count;col++)matrix_any|=matrix_high[row*cols+start+col];
      if(matrix_any||input_any)for(uint64_t col=0;col<count;col++)for(uint64_t b=0;b<batch;b++){
        uint16_t ah=matrix_high[row*cols+start+col],vh=input_high[b*cols+start+col],bits=ah|vh;
        while(bits){
          unsigned j=(unsigned)__builtin_ctz((unsigned)bits);bits&=(uint16_t)(bits-1);
          uint64_t a=tile[col][j],v=packed[(b*cols+start+col)*16+j];
          uint64_t a_high=(ah>>j)&1,v_high=(vh>>j)&1;
          u128 correction=((u128)(a_high*v+v_high*a)<<32)+((u128)(a_high*v_high)<<64);
          u128 total=((u128)high[b][j]<<64)|low[b][j];total+=correction;
          low[b][j]=(uint64_t)total;high[b][j]=(uint64_t)(total>>64);
        }
      }
    }
    for(uint64_t b=0;b<batch;b++)for(int j=0;j<16;j++){
      uint64_t index=row*16+j;
      out[b][index]=reduce128((((u128)high[b][j]<<64)|low[b][j])+(identity?identity[b][index]:0),q);
    }
  }
  return 0;
}
static int cached_a2_neon(const CRSContext *context,const uint32_t *matrix,
    const uint16_t *matrix_high,uint64_t rows,uint64_t cols,
    const uint64_t *const *vectors,uint64_t batch,
    const uint64_t *const *identity,uint64_t *const *out){
  uint32_t *packed=malloc(batch*cols*66);if(!packed)return -2;
  uint16_t *input_high=(uint16_t*)(packed+batch*cols*16);
  uint16_t input_any=pack_long_vectors(vectors,batch,cols,packed,input_high);
  int code=cached_long_packed(context,matrix,matrix_high,rows,cols,
                            packed,input_high,input_any,batch,identity,out);
  free(packed);return code;
}
#endif
int latge_context_cached_a2_dot_split(const CRSContext *context,const uint32_t *low,
    const uint16_t *high,uint64_t rows,uint64_t cols,const uint64_t *const *vectors,
    uint64_t batch,const uint64_t *const *identity,uint64_t *const *out){
#if defined(__aarch64__)
  if(context&&context->plan.n==16&&rows<=4096&&cols>0&&cols<=65536&&(batch==4||batch==5)){
    if(cols<=64)return cached_short_dot(context,low,high,rows,cols,vectors,batch,identity,out);
    return cached_a2_neon(context,low,high,rows,cols,vectors,batch,identity,out);
  }
#endif
  return cached_a2_scalar(context,low,high,rows,cols,vectors,batch,identity,out);
}

int latge_context_cached_project(const CRSContext *context,
    const uint32_t *a1,const uint16_t *a1_high,const uint32_t *a2,const uint16_t *a2_high,
    uint64_t head,uint64_t rows,uint64_t tail_cols,const uint64_t *const *values,
    const uint64_t *const *tail,uint64_t batch,uint64_t *binding,uint64_t *const *out){
  if(!context||context->plan.n!=16||head>4096||rows>4096||tail_cols>65536||batch<1||batch>16)return -1;
  uint64_t *targets[16];
  for(uint64_t b=0;b<batch;b++)targets[b]=binding+b*head*16;
  int code=latge_context_cached_a2_dot_split(context,a1,a1_high,head,rows,values,batch,NULL,targets);
  if(code)return code;
  return latge_context_cached_a2_dot_split(context,a2,a2_high,rows,tail_cols,tail,batch,values,out);
}

int latge_unpack_proof(const uint8_t *,uint64_t,int,uint64_t,uint64_t,uint64_t *);
static inline int unpack_row(const uint8_t *raw,int bits,uint64_t q,
                              uint64_t bound,uint64_t *values){
  if(bits==32&&bound){
    uint64_t cardinality=2*bound-1;
    for(int j=0;j<16;j++){
      uint32_t word;memcpy(&word,raw+4*j,4);
      if(word>=cardinality)return -1;
      int64_t value=(int64_t)word-(int64_t)bound+1;
      values[j]=(uint64_t)(value<0?value+(int64_t)q:value);
    }
  }else if(bits==33&&!bound){
    for(int j=0;j<16;j++){
      const uint8_t *at=raw+(33*j)/8;uint32_t low;memcpy(&low,at,4);
      uint64_t word=((uint64_t)low|((uint64_t)at[4]<<32))>>(j&7);
      word&=UINT64_C(0x1ffffffff);if(word>=q)return -1;
      values[j]=word;
    }
  }else if(latge_unpack_proof(raw,16,bits,q,bound,values))return -1;
  return 0;
}

int latge_projection_pack_tail(const uint64_t *const *tail,uint64_t batch,
                               uint64_t cols,uint8_t *storage){
  if(!batch||batch>16||!cols||cols>65536)return -1;
#if defined(__aarch64__)
  uint32_t *packed=(uint32_t*)(storage+8);
  uint16_t *high=(uint16_t*)(packed+batch*cols*32);
  *(uint64_t*)storage=pack_short_vectors(tail,batch,cols,packed,high);
#else
  *(uint64_t*)storage=0;
#endif
  return 0;
}

int latge_context_project_reuse(const CRSContext *context,
    const uint32_t *a1,const uint16_t *a1_high,const uint32_t *a2,const uint16_t *a2_high,
    uint64_t head,uint64_t rows,uint64_t tail_cols,const uint64_t *const *values,
    const uint64_t *const *tail,uint64_t batch,const uint8_t *packed_tail,
    void *workspace,uint64_t *binding,uint64_t *const *out,
    const uint8_t *raw,uint64_t stride,int bits,uint64_t bound,
    uint64_t input_row,uint64_t output_row){
  if(!context||context->plan.n!=16||head>4096||!rows||rows>4096||
     !tail_cols||tail_cols>65536||!batch||batch>16)return -1;
  uint64_t q=context->plan.mod.q;
  if(raw&&(bits<1||bits>56||!bound||bound>q/2||stride<rows*2*(uint64_t)bits))return -1;
  uint64_t *targets[16],*destinations[16];const uint64_t *inputs[16];
  for(uint64_t b=0;b<batch;b++){
    targets[b]=binding+b*head*16;destinations[b]=out[b]+output_row*16;
    if(!raw)inputs[b]=values[b]+input_row*16;
  }
  if(!raw)values=inputs;
  out=destinations;
#if defined(__aarch64__)
  if((batch==4||batch==5)&&tail_cols<=64){
    const uint32_t *tail_low=(const uint32_t*)(packed_tail+8);
    const uint16_t *tail_high=(const uint16_t*)(tail_low+batch*tail_cols*32);
    uint16_t tail_any=(uint16_t)*(const uint64_t*)packed_tail;
    uint32_t *packed=workspace;
    uint16_t *high=(uint16_t*)(packed+batch*rows*16),input_any=0;
    if(raw){
      ResponseProjection response={raw,stride,rows,bits,bound,packed,high,0};
      int code=cached_short_packed(context,a2,a2_high,rows,tail_cols,
                     tail_low,tail_high,tail_any,batch,NULL,out,&response);
      if(code)return code;
      input_any=response.any;
    }else input_any=pack_long_vectors(values,batch,rows,packed,high);
    int code=cached_long_packed(context,a1,a1_high,head,rows,
                               packed,high,input_any,batch,NULL,targets);
    if(code||raw)return code;
    return cached_short_packed(context,a2,a2_high,rows,tail_cols,
                   tail_low,tail_high,tail_any,batch,values,out,NULL);
  }
#endif
  const uint64_t *decoded[16];
  if(raw){
    for(uint64_t b=0;b<batch;b++){
      uint64_t *slots=(uint64_t*)workspace+b*rows*16;decoded[b]=slots;
      for(uint64_t row=0;row<rows;row++){
        if(unpack_row(raw+b*stride+row*2*bits,bits,q,bound,slots+row*16))return -1;
        transform16(&context->plan,slots+row*16,0);
      }
    }
    values=decoded;
  }
  return latge_context_cached_project(context,a1,a1_high,a2,a2_high,head,rows,
                                     tail_cols,values,tail,batch,binding,out);
}

typedef struct {
  const uint32_t *a1;const uint16_t *h1;
  const uint32_t *a2;const uint16_t *h2;
  uint64_t start,rows;
} ProjectionBlock;
int latge_projection_group(const CRSContext *context,const ProjectionBlock *blocks,
    uint64_t count,uint64_t first,uint64_t step,uint64_t head,uint64_t tail_cols,
    uint64_t batch,const uint64_t *const *values,const uint64_t *const *tail,
    uint64_t *const *out,const uint8_t *packed_tail,void *scratch,
    uint64_t *binding,uint64_t *sum,const uint8_t *raw,uint64_t stride,int bits,uint64_t bound){
  if(!context||!step||first>=count||head>4096||!batch||batch>16)return -1;
  uint64_t size=batch*head*16,q=context->plan.mod.q;
  memset(sum,0,size*8);
  for(uint64_t i=first;i<count;i+=step){
    const ProjectionBlock *b=blocks+i;
    const uint8_t *response=raw?raw+(head+b->start)*2*bits:NULL;
    int code=latge_context_project_reuse(context,b->a1,b->h1,b->a2,b->h2,
        head,b->rows,tail_cols,values,tail,batch,packed_tail,scratch,binding,out,
        response,stride,bits,bound,head+b->start,b->start);
    if(code)return code;
    for(uint64_t j=0;j<size;j++){
      uint64_t value=sum[j]+binding[j];sum[j]=value>=q?value-q:value;
    }
  }
  return 0;
}

int latge_context_cached_a2_dot(const CRSContext *context,const uint32_t *low,
    const uint16_t *high,uint64_t rows,uint64_t cols,const uint64_t *vectors,
    uint64_t batch,const uint64_t *identity,uint64_t *out){
  if(batch<1||batch>16)return -1;
  const uint64_t *v[16],*id[16];uint64_t *target[16];
  for(uint64_t b=0;b<batch;b++){
    v[b]=vectors+b*cols*16;id[b]=identity?identity+b*rows*16:NULL;target[b]=out+b*rows*16;
  }
  return latge_context_cached_a2_dot_split(context,low,high,rows,cols,v,batch,identity?id:NULL,target);
}
int latge_context_transform(const CRSContext *context,const uint64_t *input,uint64_t count,int inverse,uint64_t *out){
  if(!context)return -1;int n=context->plan.n;
  for(uint64_t i=0;i<count;i++){
    memcpy(out+i*n,input+i*n,n*8);transform(&context->plan,out+i*n,inverse);
  }
  return 0;
}
int latge_commitment_encode(const CRSContext *context,const uint64_t *left,
    const uint64_t *right,uint64_t rows,int width,uint8_t *out){
  if(!context||width<1||width>8)return -1;
  const Plan *plan=&context->plan;int n=plan->n;uint64_t q=plan->mod.q,values[256];
  for(uint64_t row=0;row<rows;row++){
    for(int j=0;j<n;j++){
      uint64_t a=left[row*n+j],b=right[row*n+j];if(a>=q||b>=q)return -1;
      uint64_t sum=a+b;values[j]=sum>=q?sum-q:sum;
    }
    transform(plan,values,1);
    for(int j=0;j<n;j++)for(int byte=0;byte<width;byte++)
      out[(row*n+j)*width+byte]=(uint8_t)(values[j]>>(8*byte));
  }
  return 0;
}
int latge_context_signed_transform(const CRSContext *context,const int64_t *input,uint64_t count,uint64_t *out){
  if(!context)return -1;int n=context->plan.n;int64_t q=(int64_t)context->plan.mod.q;
  for(uint64_t i=0;i<count;i++){
    for(int j=0;j<n;j++){
      int64_t value=input[i*n+j];
      if(value<=-q||value>=q)return -1;
      out[i*n+j]=(uint64_t)(value<0?value+q:value);
    }
    transform(&context->plan,out+i*n,0);
  }
  return 0;
}
int latge_trinary_bytes_transform(const CRSContext *context,const uint8_t *raw,uint64_t bytes,
                                  uint64_t rows,uint64_t *out){
  if(!context)return -1;const Plan *p=&context->plan;uint64_t q=p->mod.q;
  if(rows>UINT64_MAX/(uint64_t)p->n)return -1;
  uint64_t count=rows*p->n;if(bytes!=(count+1)/2)return -1;
  for(uint64_t row=0;row<rows;row++){
    for(int j=0;j<p->n;j++){
      uint64_t i=row*p->n+j;unsigned nibble=(raw[i/2]>>((i&1)*4))&15;
      int value=(int)((UINT32_C(0xA815)>>nibble)&3)-1;
      out[i]=value<0?q-1:(uint64_t)value;
    }
    transform(p,out+row*p->n,0);
  }
  return 0;
}
int latge_bounded_bytes_transform(const CRSContext *context,const uint8_t *raw,uint64_t bytes,
                                  uint64_t bound,uint64_t rows,uint64_t *produced,uint64_t *out){
  if(!context||!produced)return -1;const Plan *p=&context->plan;uint64_t q=p->mod.q;
  if(bound>q/2||rows>UINT64_MAX/(uint64_t)p->n)return -1;
  uint64_t count=rows*p->n,done=*produced,span=2*bound+1;
  unsigned width=span<((uint64_t)1<<32)?4:8;
  if(done>count||bytes%width)return -1;
  if(p->n==16&&span==UINT32_MAX){
    for(uint64_t position=0;position<bytes&&done<count;position+=4){
      uint32_t word;memcpy(&word,raw+position,4);if(word==UINT32_MAX)continue;
      uint64_t value=word;out[done++]=value<bound?q+value-bound:value-bound;
      if(!(done&15))transform16(p,out+done-16,0);
    }
    *produced=done;return 0;
  }
  u128 cardinality=(u128)1<<(8*width),limit=(cardinality/span)*span;
  for(uint64_t position=0;position<bytes&&done<count;position+=width){
    uint64_t word;if(width==4){uint32_t x;memcpy(&x,raw+position,4);word=x;}
    else memcpy(&word,raw+position,8);
    if((u128)word>=limit)continue;
    uint64_t value=word%span;
    out[done++]=value<bound?q+value-bound:value-bound;
    if(done%(uint64_t)p->n==0)transform(p,out+done-p->n,0);
  }
  *produced=done;return 0;
}
int latge_uniform_challenge(const CRSContext *context,const uint8_t *raw,
    uint64_t bytes,uint64_t count,int transform_rows,uint64_t *produced,uint64_t *out){
  if(!context||!produced)return -1;
  const Plan *p=&context->plan;uint64_t q=p->mod.q,done=*produced;
  unsigned width=1;while(q>>(8*width))width++;
  if(done>count||bytes%width||(transform_rows&&count%(uint64_t)p->n))return -1;
  uint64_t limit=(((uint64_t)1<<(8*width))/q)*q;
  if(q==UINT64_C(4294967969)){
    for(uint64_t position=0;position<bytes&&done<count;position+=5){
      uint32_t low;memcpy(&low,raw+position,4);uint64_t high=raw[position+4];
      if(((high<<32)|low)>=limit)continue;
      int64_t value=(int64_t)low-INT64_C(673)*(int64_t)high;
      out[done++]=(uint64_t)(value<0?value+INT64_C(4294967969):value);
      if(transform_rows&&done%(uint64_t)p->n==0)transform(p,out+done-p->n,0);
    }
    *produced=done;return 0;
  }
  for(uint64_t position=0;position<bytes&&done<count;position+=width){
    uint64_t word=0;
    for(unsigned j=0;j<width;j++)word|=(uint64_t)raw[position+j]<<(8*j);
    if(word>=limit)continue;
    out[done++]=word%q;
    if(transform_rows&&done%(uint64_t)p->n==0)transform(p,out+done-p->n,0);
  }
  *produced=done;return 0;
}
int latge_context_dot_split(const CRSContext *context,int kind,uint64_t first_row,uint64_t rows,uint64_t first_col,uint64_t cols,const uint64_t *const *vectors,uint64_t batch,const uint64_t *const *identity,uint64_t *const *out){
  if(!context||kind<0||kind>1)return -1;
  return crs_dot_split(&context->prefix[kind],&context->plan,first_row,rows,first_col,cols,vectors,batch,identity,out);
}
int latge_context_dot(const CRSContext *context,int kind,uint64_t first_row,uint64_t rows,uint64_t first_col,uint64_t cols,const uint64_t *vectors,uint64_t batch,const uint64_t *identity,uint64_t *out){
  if(!context||kind<0||kind>1)return -1;
  return crs_dot_prepared(&context->prefix[kind],&context->plan,first_row,rows,first_col,cols,vectors,batch,identity,out);
}

int latge_subtract_challenge_products(const CRSContext *context,uint64_t **values,
    const uint64_t *challenges,const uint64_t *target,uint64_t rows,uint64_t batch){
  if(!context||!values||!challenges||!target||batch<1||batch>16)return -1;
  const Plan *p=&context->plan;uint64_t q=p->mod.q;int n=p->n;
  for(uint64_t row=0;row<rows;row++)for(int j=0;j<n;j++){
    uint64_t index=row*n+j,t=target[index];
    for(uint64_t lane=0;lane<batch;lane++){
      uint64_t product=mulmod(challenges[lane*n+j],t,&p->mod);
      uint64_t difference=values[lane][index]+q-product;
      values[lane][index]=difference>=q?difference-q:difference;
    }
  }
  return 0;
}

int latge_response(const CRSContext *context,const uint64_t *r,const uint64_t *y,
                   const uint64_t *challenge,uint64_t count,uint64_t bound,uint64_t *out){
  if(!context||!bound)return -1;
  const Plan *p=&context->plan;uint64_t q=p->mod.q;int n=p->n;
  for(int j=0;j<n;j++)if(challenge[j]>=q)return -1;
  for(uint64_t row=0;row<count;row++){
    uint64_t values[256];
    for(int j=0;j<n;j++){
      uint64_t index=row*n+j;if(r[index]>=q||y[index]>=q)return -1;
      uint64_t value=mulmod(r[index],challenge[j],&p->mod)+y[index];
      values[j]=value>=q?value-q:value;
    }
    transform(p,values,1);
    for(int j=0;j<n;j++)if((values[j]<=q/2?values[j]:q-values[j])>=bound)return 1;
    memcpy(out+row*n,values,n*8);
  }
  return 0;
}
int latge_responses(const CRSContext *context,const uint64_t *r,
    uint64_t *const *y,const uint64_t *challenges,uint64_t count,
    uint64_t bound,uint64_t lanes){
  if(!context||!bound||!lanes||lanes>16)return -1;
  const Plan *p=&context->plan;uint64_t q=p->mod.q;int n=p->n;
  for(uint64_t lane=0;lane<lanes;lane++)for(int j=0;j<n;j++)
    if(challenges[lane*n+j]>=q)return -1;
  for(uint64_t row=0;row<count;row++){
    const uint64_t *opening=r+row*n;
    for(int j=0;j<n;j++)if(opening[j]>=q)return -1;
    for(uint64_t lane=0;lane<lanes;lane++){
      uint64_t *mask=y[lane]+row*n,values[256];
      for(int j=0;j<n;j++){
        if(mask[j]>=q)return -1;
        uint64_t value=mulmod(opening[j],challenges[lane*n+j],&p->mod)+mask[j];
        values[j]=value>=q?value-q:value;
      }
      transform(p,values,1);
      for(int j=0;j<n;j++)if((values[j]<=q/2?values[j]:q-values[j])>=bound)return 1;
      memcpy(mask,values,n*8);
    }
  }
  return 0;
}

int latge_decode_response(const CRSContext *context,const uint8_t *bytes,
                          uint64_t count,int width,uint64_t bound,uint64_t *out){
  if(!context||width<1||width>8||!bound)return -1;
  const Plan *p=&context->plan;uint64_t q=p->mod.q;int n=p->n;
  for(uint64_t row=0;row<count;row++){
    uint64_t values[256];
    for(int j=0;j<n;j++){
      uint64_t value=0;for(int b=0;b<width;b++)value|=(uint64_t)bytes[(row*n+j)*width+b]<<(8*b);
      if(value>=q)return -1;
      if((value<=q/2?value:q-value)>=bound)return 1;
      values[j]=value;
    }
    transform(p,values,0);memcpy(out+row*n,values,n*8);
  }
  return 0;
}

int latge_decode_transform(const CRSContext *context,const uint8_t *bytes,
                           uint64_t count,int width,uint64_t *out){
  if(!context||width<1||width>8)return -1;
  const Plan *p=&context->plan;uint64_t q=p->mod.q;int n=p->n;
  for(uint64_t row=0;row<count;row++){
    uint64_t *values=out+row*n;
    for(int j=0;j<n;j++){
      uint64_t value=0;
      for(int b=0;b<width;b++)value|=(uint64_t)bytes[(row*n+j)*width+b]<<(8*b);
      if(value>=q)return -1;
      values[j]=value;
    }
    transform(p,values,0);
  }
  return 0;
}

static uint64_t addmod(uint64_t a,uint64_t b,uint64_t q){uint64_t s=a+b;return s>=q?s-q:s;}
static uint64_t submod(uint64_t a,uint64_t b,uint64_t q){return a>=b?a-b:a+q-b;}
static uint64_t triplemod(uint64_t a,uint64_t q){return addmod(addmod(a,a,q),a,q);}
int latge_slot_dot(const uint64_t *left,const uint64_t *const *right,
                   uint64_t rows,int n,uint64_t q,uint64_t batch,uint64_t *out){
  if(!valid(n,q)||rows>((uint64_t)1<<20)||batch<1||batch>16)return -1;
  u128 acc[16][256]={{0}};
  for(uint64_t i=0;i<rows;i++)for(int j=0;j<n;j++){
    uint64_t x=left[i*n+j];
    for(uint64_t b=0;b<batch;b++)acc[b][j]+=(u128)x*right[b][i*n+j];
  }
  for(uint64_t b=0;b<batch;b++)for(int j=0;j<n;j++)out[b*n+j]=reduce128(acc[b][j],q);
  return 0;
}
int latge_slot_multi_dot(const uint64_t *const *left,const uint64_t *const *right,
                         uint64_t rows,int n,uint64_t q,uint64_t groups,uint64_t batch,uint64_t *out){
  if(!valid(n,q)||rows>((uint64_t)1<<20)||groups<1||groups>16||batch<1||batch>16)return -1;
  size_t count=groups*batch*n;u128 *acc=calloc(count,sizeof(u128));if(!acc)return -2;
  for(uint64_t i=0;i<rows;i++)for(int j=0;j<n;j++){
    uint64_t index=i*n+j;
    for(uint64_t b=0;b<batch;b++){
      uint64_t y=right[b][index];
      for(uint64_t g=0;g<groups;g++)acc[(g*batch+b)*n+j]+=(u128)left[g][index]*y;
    }
  }
  for(size_t i=0;i<count;i++)out[i]=reduce128(acc[i],q);
  free(acc);return 0;
}
int latge_slot_adjoint_product_dot(const uint64_t *const *source,const uint64_t *const *right,
 int n,int e,uint64_t q,uint64_t weight,uint64_t groups,uint64_t batch,uint64_t *out){
  if(!valid(e,q)||n<e||n>1024||n%e||groups<1||groups>16||batch<1||batch>16||weight>=q)return -1;
  size_t count=groups*batch*e;
  int64_t *collapsed=calloc(batch*n*e,sizeof(int64_t));if(!collapsed)return -2;
  for(int row=0;row<n;row++)for(int col=0;col<n;col++){
    int sum=row+col,slot=col%e;uint64_t index=(uint64_t)row*n+col;
    int sign=sum<n?1:-1,src=sum<n?sum:sum-n;
    for(uint64_t b=0;b<batch;b++){
      collapsed[(b*n+src)*e+slot]+=sign*(int64_t)right[b][index];
    }
  }
  __int128 acc[count];memset(acc,0,sizeof(acc));Modulus mod=modulus(q);
  for(int src=0;src<n;src++)for(int slot=0;slot<e;slot++)
    for(uint64_t b=0;b<batch;b++){
      int64_t value=collapsed[(b*n+src)*e+slot];
      for(uint64_t g=0;g<groups;g++)acc[(g*batch+b)*e+slot]+=(__int128)source[g][src]*value;
    }
  for(size_t i=0;i<count;i++){
    int64_t value=(int64_t)(acc[i]%q);if(value<0)value+=q;
    out[i]=mulmod((uint64_t)value,weight,&mod);
  }
  free(collapsed);return 0;
}
int latge_expansion_sums(int kind,const uint64_t *alpha,const uint64_t *message,
                         const uint64_t *mask,const uint64_t *other_message,
                         const uint64_t *other_mask,const uint64_t *output_mask,
                         const uint64_t *indicator,uint64_t rows,int n,uint64_t q,uint64_t *out){
  if(!valid(n,q)||rows>4096||kind<0||kind>2)return -1;
  if((kind==0&&(!other_message||!other_mask||!output_mask))||(kind==2&&!indicator))return -1;
  Modulus mod=modulus(q);u128 acc[3][256]={{0}};
  for(uint64_t i=0;i<rows;i++)for(int j=0;j<n;j++){
    uint64_t t=i*n+j,a=alpha[t],m=message[t],y=mask[t],v0=0,v1,v2;
    if(kind==0){
      v1=mulmod(y,other_mask[t],&mod);
      v2=submod(addmod(mulmod(y,other_message[t],&mod),mulmod(m,other_mask[t],&mod),q),output_mask[t],q);
    }else if(kind==1){
      uint64_t yy=mulmod(y,y,&mod),mm=mulmod(m,m,&mod);
      v0=mulmod(yy,y,&mod);v1=triplemod(mulmod(yy,m,&mod),q);
      v2=submod(triplemod(mulmod(y,mm,&mod),q),y,q);
    }else{
      v1=mulmod(y,y,&mod);uint64_t ym=mulmod(y,m,&mod);
      v2=submod(addmod(ym,ym,q),indicator[t]?y:0,q);
    }
    acc[0][j]+=(u128)a*v0;acc[1][j]+=(u128)a*v1;acc[2][j]+=(u128)a*v2;
  }
  for(int k=0;k<3;k++)for(int j=0;j<n;j++)out[k*n+j]=reduce128(acc[k][j],q);
  return 0;
}
int latge_homogeneous_sums(int kind,const uint64_t *alpha,const uint64_t *x,
                           const uint64_t *y,const uint64_t *z,const uint64_t *indicator,
                           const uint64_t *challenge,uint64_t rows,int n,uint64_t q,uint64_t *out){
  if(!valid(n,q)||rows>4096||kind<0||kind>2)return -1;
  if((kind==0&&(!y||!z))||(kind==2&&!indicator))return -1;
  Modulus mod=modulus(q);u128 acc[256]={0};uint64_t square[256];
  for(int j=0;j<n;j++)square[j]=mulmod(challenge[j],challenge[j],&mod);
  for(uint64_t i=0;i<rows;i++)for(int j=0;j<n;j++){
    uint64_t t=i*n+j,v;
    if(kind==0)v=addmod(mulmod(challenge[j],mulmod(x[t],y[t],&mod),&mod),mulmod(square[j],z[t],&mod),q);
    else if(kind==1)v=submod(mulmod(square[j],x[t],&mod),mulmod(mulmod(x[t],x[t],&mod),x[t],&mod),q);
    else v=addmod(mulmod(challenge[j],mulmod(x[t],x[t],&mod),&mod),indicator[t]?mulmod(square[j],x[t],&mod):0,q);
    acc[j]+=(u128)alpha[t]*v;
  }
  for(int j=0;j<n;j++)out[j]=reduce128(acc[j],q);
  return 0;
}

int latge_multi_expansion_sums(int kind,const uint64_t *alpha,const uint64_t *message,
 const uint64_t *other_message,const uint64_t *const *mask,const uint64_t *const *other_mask,
 const uint64_t *const *output_mask,const uint64_t *indicator,uint64_t rows,uint64_t other_rows,int n,uint64_t q,
 uint64_t lanes,uint64_t *out){
  if(!valid(n,q)||rows>4096||kind<0||kind>2||lanes<1||lanes>16)return -1;
  if((kind==0&&(!other_message||!other_mask||!output_mask||(rows&&(!other_rows||other_rows>rows))))||(kind==2&&!indicator))return -1;
  if(kind==0){
    u128 positive[lanes][2][n],negative[lanes][n];
    memset(positive,0,sizeof(positive));memset(negative,0,sizeof(negative));
    for(uint64_t i=0;i<rows;i++){
      uint64_t right_base=(i%other_rows)*n;
      for(int j=0;j<n;j++){
        uint64_t t=i*n+j,u=right_base+j,a=alpha[t],m=message[t],om=other_message[u];
        u128 am=(u128)a*m;
        for(uint64_t lane=0;lane<lanes;lane++){
          uint64_t y=mask[lane][t],oy=other_mask[lane][u];u128 ay=(u128)a*y;
          positive[lane][0][j]+=ay*oy;
          positive[lane][1][j]+=ay*om+am*oy;
          negative[lane][j]+=(u128)a*output_mask[lane][t];
        }
      }
    }
    for(uint64_t lane=0;lane<lanes;lane++)for(int j=0;j<n;j++){
      out[(lane*3)*n+j]=0;
      out[(lane*3+1)*n+j]=reduce128(positive[lane][0][j],q);
      out[(lane*3+2)*n+j]=submod(reduce128(positive[lane][1][j],q),
                                                reduce128(negative[lane][j],q),q);
    }
    return 0;
  }
  Modulus mod=modulus(q);u128 acc[lanes][3][n];memset(acc,0,sizeof(acc));
  for(uint64_t i=0;i<rows;i++){
   uint64_t right_base=kind==0?(i%other_rows)*n:0;
   for(int j=0;j<n;j++){
    uint64_t t=i*n+j,u=right_base+j,a=alpha[t],m=message[t];
    uint64_t mm=kind==1?mulmod(m,m,&mod):0;
    for(uint64_t lane=0;lane<lanes;lane++){
      uint64_t y=mask[lane][t],v0=0,v1,v2;
      if(kind==0){
        v1=mulmod(y,other_mask[lane][u],&mod);
        v2=submod(addmod(mulmod(y,other_message[u],&mod),mulmod(m,other_mask[lane][u],&mod),q),output_mask[lane][t],q);
      }else if(kind==1){
        uint64_t yy=mulmod(y,y,&mod);v0=mulmod(yy,y,&mod);
        v1=triplemod(mulmod(yy,m,&mod),q);v2=submod(triplemod(mulmod(y,mm,&mod),q),y,q);
      }else{
        v1=mulmod(y,y,&mod);uint64_t ym=mulmod(y,m,&mod);
        v2=submod(addmod(ym,ym,q),indicator[t]?y:0,q);
      }
      acc[lane][0][j]+=(u128)a*v0;acc[lane][1][j]+=(u128)a*v1;acc[lane][2][j]+=(u128)a*v2;
    }
   }
  }
  for(uint64_t lane=0;lane<lanes;lane++)for(int k=0;k<3;k++)for(int j=0;j<n;j++)
    out[(lane*3+k)*n+j]=reduce128(acc[lane][k][j],q);
  return 0;
}
static int multi_homogeneous_sums(int kind,const uint64_t *alpha,const uint64_t *const *x,
 const uint64_t *const *y,const uint64_t *const *z,const uint64_t *indicator,const uint64_t *challenge,
 uint64_t rows,uint64_t other_rows,int n,uint64_t q,uint64_t lanes,uint64_t *out,const uint64_t *output_target){
  if(!valid(n,q)||rows>4096||kind<0||kind>2||lanes<1||lanes>16)return -1;
  if((kind==0&&(!y||!z||(rows&&(!other_rows||other_rows>rows))))||(kind==2&&!indicator))return -1;
  Modulus mod=modulus(q);u128 acc[lanes][n];memset(acc,0,sizeof(acc));uint64_t square[lanes][n];
  for(uint64_t lane=0;lane<lanes;lane++)for(int j=0;j<n;j++)
    square[lane][j]=mulmod(challenge[lane*n+j],challenge[lane*n+j],&mod);
  if(kind==0){
    u128 outputs[lanes][n],target[n];memset(outputs,0,sizeof(outputs));memset(target,0,sizeof(target));
    for(uint64_t i=0;i<rows;i++){
      uint64_t right_base=(i%other_rows)*n;
      for(int j=0;j<n;j++){
        uint64_t t=i*n+j,u=right_base+j,a=alpha[t];
        if(output_target)target[j]+=(u128)a*output_target[t];
        for(uint64_t lane=0;lane<lanes;lane++){
          acc[lane][j]+=(u128)a*x[lane][t]*y[lane][u];
          outputs[lane][j]+=(u128)a*z[lane][t];
        }
      }
    }
    uint64_t reduced_target[n];
    if(output_target)for(int j=0;j<n;j++)reduced_target[j]=reduce128(target[j],q);
    for(uint64_t lane=0;lane<lanes;lane++)for(int j=0;j<n;j++){
      uint64_t output_sum=reduce128(outputs[lane][j],q);
      if(output_target)output_sum=submod(output_sum,
          mulmod(challenge[lane*n+j],reduced_target[j],&mod),q);
      out[lane*n+j]=addmod(mulmod(challenge[lane*n+j],reduce128(acc[lane][j],q),&mod),
                          mulmod(square[lane][j],output_sum,&mod),q);
    }
    return 0;
  }
  for(uint64_t i=0;i<rows;i++){
   uint64_t right_base=kind==0?(i%other_rows)*n:0;
   for(int j=0;j<n;j++){
    uint64_t t=i*n+j,u=right_base+j,a=alpha[t];
    for(uint64_t lane=0;lane<lanes;lane++){
      uint64_t value=x[lane][t],c=challenge[lane*n+j],sq=square[lane][j],v;
      if(kind==0)v=addmod(mulmod(c,mulmod(value,y[lane][u],&mod),&mod),mulmod(sq,z[lane][t],&mod),q);
      else if(kind==1)v=submod(mulmod(sq,value,&mod),mulmod(mulmod(value,value,&mod),value,&mod),q);
      else v=addmod(mulmod(c,mulmod(value,value,&mod),&mod),indicator[t]?mulmod(sq,value,&mod):0,q);
      acc[lane][j]+=(u128)a*v;
    }
   }
  }
  for(uint64_t lane=0;lane<lanes;lane++)for(int j=0;j<n;j++)out[lane*n+j]=reduce128(acc[lane][j],q);
  return 0;
}

int latge_multi_homogeneous_sums(int kind,const uint64_t *alpha,const uint64_t *const *x,
 const uint64_t *const *y,const uint64_t *const *z,const uint64_t *indicator,const uint64_t *challenge,
 uint64_t rows,uint64_t other_rows,int n,uint64_t q,uint64_t lanes,uint64_t *out){
  return multi_homogeneous_sums(kind,alpha,x,y,z,indicator,challenge,rows,other_rows,n,q,lanes,out,NULL);
}
int latge_multi_homogeneous_affine_outputs(int kind,const uint64_t *alpha,const uint64_t *const *x,
 const uint64_t *const *y,const uint64_t *const *z,const uint64_t *indicator,const uint64_t *challenge,
 uint64_t rows,uint64_t other_rows,int n,uint64_t q,uint64_t lanes,uint64_t *out,const uint64_t *target){
  return multi_homogeneous_sums(kind,alpha,x,y,z,indicator,challenge,rows,other_rows,n,q,lanes,out,target);
}

int latge_encode_coefficients(const uint64_t *values,uint64_t count,
                              int width,uint64_t q,uint8_t *bytes){
  if(width<1||width>8||q<2)return -1;
  for(uint64_t i=0;i<count;i++){
    uint64_t value=values[i];if(value>=q)return -1;
    for(int j=0;j<width;j++){bytes[i*width+j]=(uint8_t)value;value>>=8;}
    if(value)return -1;
  }
  return 0;
}
int latge_decode_coefficients(const uint8_t *bytes,uint64_t count,
                              int width,uint64_t q,uint64_t *values){
  if(width<1||width>8||q<2)return -1;
  for(uint64_t i=0;i<count;i++){
    uint64_t value=0;
    for(int j=0;j<width;j++)value|=(uint64_t)bytes[i*width+j]<<(8*j);
    if(value>=q)return -1;
    values[i]=value;
  }
  return 0;
}

int latge_pack_proof(const uint64_t *values,uint64_t count,int bits,
                    uint64_t q,uint64_t bound,uint8_t *bytes){
  if(bits<1||bits>56||q<3||q>=(1ULL<<56)||bound>q/2)return -1;
  uint64_t acc=0,pos=0;int pending=0;
  for(uint64_t i=0;i<count;i++){
    uint64_t value=values[i];if(value>=q)return -1;
    if(bound){
      int64_t z=value>q/2?(int64_t)value-(int64_t)q:(int64_t)value;
      if(z<=-(int64_t)bound||z>=(int64_t)bound)return -1;
      value=(uint64_t)(z+(int64_t)bound-1);
    }
    if(value>>bits)return -1;
    acc|=value<<pending;pending+=bits;
    while(pending>=8){bytes[pos++]=(uint8_t)acc;acc>>=8;pending-=8;}
  }
  if(pending)bytes[pos]=(uint8_t)acc;
  return 0;
}
int latge_unpack_proof(const uint8_t *bytes,uint64_t count,int bits,
                      uint64_t q,uint64_t bound,uint64_t *values){
  if(bits<1||bits>56||q<3||q>=(1ULL<<56)||bound>q/2)return -1;
  uint64_t acc=0,pos=0,mask=(1ULL<<bits)-1;
  uint64_t cardinality=bound?2*bound-1:q;int pending=0;
  for(uint64_t i=0;i<count;i++){
    while(pending<bits){acc|=(uint64_t)bytes[pos++]<<pending;pending+=8;}
    uint64_t value=acc&mask;acc>>=bits;pending-=bits;
    if(value>=cardinality)return -1;
    if(bound){
      int64_t z=(int64_t)value-(int64_t)bound+1;
      value=z<0?(uint64_t)(z+(int64_t)q):(uint64_t)z;
    }
    values[i]=value;
  }
  return acc?-1:0;
}

int latge_unpack_transform(const CRSContext *context,const uint8_t *bytes,
    uint64_t rows,int bits,uint64_t bound,uint64_t *slots,uint8_t *hash_bytes){
  if(!context||context->plan.n!=16||bits<1||bits>56||16*bits%8)return -1;
  uint64_t q=context->plan.mod.q,stride=2*bits;
  if(bound>q/2)return -1;
  unsigned hash_width=1;while(q>>(8*hash_width))hash_width++;
  for(uint64_t row=0;row<rows;row++){
    uint64_t *values=slots+row*16;
    const uint8_t *raw=bytes+row*stride;
    if(unpack_row(raw,bits,q,bound,values))return -1;
    if(hash_bytes)for(int j=0;j<16;j++){
      uint64_t value=values[j];uint8_t *at=hash_bytes+(row*16+j)*hash_width;
      for(unsigned b=0;b<hash_width;b++){at[b]=(uint8_t)value;value>>=8;}
    }
    transform16(&context->plan,values,0);
  }
  return 0;
}

int latge_unpack_transform_lanes(const CRSContext *context,const uint8_t *bytes,
    uint64_t stride,uint64_t rows,int bits,uint64_t bound,uint64_t lanes,uint64_t *slots){
  if(!context||context->plan.n!=16||!lanes||lanes>16||rows>4096||
      bits<1||bits>56||stride<rows*2*(uint64_t)bits)return -1;
  for(uint64_t lane=0;lane<lanes;lane++){
    int code=latge_unpack_transform(context,bytes+lane*stride,rows,bits,bound,
                                    slots+lane*rows*16,NULL);
    if(code)return code;
  }
  return 0;
}

#include "ring_native.c"
int latge_original_transform(const uint64_t *in,uint64_t count,int inverse,uint64_t *out){
  for(uint64_t row=0;row<count;row++){
    if(inverse)table_inverse(in+128*row,(int64_t *)(out+128*row),table_powers,table_untwist);
    else {
      int64_t tmp[128];table_forward((const int64_t *)(in+128*row),tmp,table_powers);
      for(int j=0;j<128;j++)out[128*row+j]=tmp[j]<0?(uint64_t)(tmp[j]+(int64_t)TABLE_Q):(uint64_t)tmp[j];
    }
  }
  return 0;
}
static void original_field_mul(const uint64_t *a,const uint64_t *b,int slot,uint64_t *out){
  __int128 acc[15]={0};
  for(int i=0;i<8;i++)for(int j=0;j<8;j++)acc[i+j]+=(__int128)a[i]*b[j];
  for(int i=0;i<8;i++){
    uint64_t v=table_reduce(acc[i]);
    if(i<7){v+=table_mul(table_powers[2*slot+1],table_reduce(acc[i+8]));if(v>=TABLE_Q)v-=TABLE_Q;}
    out[i]=v;
  }
}
int latge_original_mul(const uint64_t *a,const uint64_t *b,uint64_t count,
                       int a_single,int b_single,int sum,uint64_t *out){
  if(sum)memset(out,0,128*sizeof(uint64_t));
  for(uint64_t row=0;row<count;row++)for(int slot=0;slot<16;slot++){
    uint64_t tmp[8];
    original_field_mul(a+(a_single?0:128*row)+8*slot,b+(b_single?0:128*row)+8*slot,slot,tmp);
    for(int j=0;j<8;j++){
      uint64_t index=(sum?0:128*row)+8*slot+j;
      if(sum){uint64_t v=out[index]+tmp[j];out[index]=v>=TABLE_Q?v-TABLE_Q:v;}
      else out[index]=tmp[j];
    }
  }
  return 0;
}
int latge_original_dot(const uint64_t *matrix,const uint64_t *vectors,
    uint64_t rows,uint64_t cols,uint64_t lanes,uint64_t *out){
  if(rows>4096||cols>65536||lanes>2)return -1;
  for(uint64_t row=0;row<rows;row++)for(uint64_t lane=0;lane<lanes;lane++)for(int slot=0;slot<16;slot++){
    __int128 acc[15]={0};
    for(uint64_t col=0;col<cols;col++){
      const uint64_t *a=matrix+(row*cols+col)*128+8*slot;
      const uint64_t *b=vectors+(lane*cols+col)*128+8*slot;
      for(int i=0;i<8;i++)for(int j=0;j<8;j++)acc[i+j]+=(__int128)a[i]*b[j];
    }
    for(int j=0;j<8;j++){
      uint64_t v=table_reduce(acc[j]);
      if(j<7){v+=table_mul(table_powers[2*slot+1],table_reduce(acc[j+8]));if(v>=TABLE_Q)v-=TABLE_Q;}
      out[(lane*rows+row)*128+8*slot+j]=v;
    }
  }
  return 0;
}
