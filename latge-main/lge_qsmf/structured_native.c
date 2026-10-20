#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <stdlib.h>

static uint64_t reduce_signed128(__int128 value,uint64_t q) {
    if(q!=UINT64_C(4294967969)) {
        int64_t reduced=(int64_t)(value%q);
        return reduced<0?(uint64_t)(reduced+q):(uint64_t)reduced;
    }
    int negative=value<0;
    unsigned __int128 magnitude=negative?-(unsigned __int128)value:(unsigned __int128)value;
    int64_t folded=(int64_t)(uint32_t)magnitude
        -INT64_C(673)*(int64_t)(uint32_t)(magnitude>>32)
        +INT64_C(452929)*(int64_t)(uint32_t)(magnitude>>64)
        -INT64_C(304821217)*(int64_t)(uint32_t)(magnitude>>96);
    folded=(int64_t)(uint32_t)folded-INT64_C(673)*(folded>>32);
    folded=(int64_t)(uint32_t)folded-INT64_C(673)*(folded>>32);
    uint64_t result=(uint64_t)(folded<0?folded+(int64_t)q:folded);
    return negative&&result?q-result:result;
}

int latge_adjoint_digit_dot(const uint64_t *const *source,
        const uint64_t *const *right,const uint64_t *weights,size_t digits,
        size_t n,size_t e,uint64_t q,size_t groups,size_t batch,uint64_t *out) {
    if (!source || !right || !weights || !out || !digits || digits>64
            || e<4 || e>256 || (e&(e-1)) || n<e || n>1024 || n%e
            || q<=2 || q>=((uint64_t)1<<33) || !groups || groups>16
            || !batch || batch>16 || digits*n*n>((size_t)1<<20)) return -1;
    for(size_t d=0;d<digits;d++) if(weights[d]>=q) return -1;
    size_t count=groups*batch*e;
    int64_t *collapsed=malloc(batch*n*e*sizeof(int64_t));
    __int128 *weighted=calloc(batch*n*e,sizeof(__int128));
    __int128 *acc=calloc(count,sizeof(__int128));
    if(!collapsed || !weighted || !acc){free(collapsed);free(weighted);free(acc);return -2;}
    for(size_t d=0;d<digits;d++) {
        memset(collapsed,0,batch*n*e*sizeof(int64_t));
        for(size_t b=0;b<batch;b++) {
            const uint64_t *values=right[b]+d*n*n;
            int64_t *sums=collapsed+b*n*e;
            for(size_t row=0;row<n;row++) {
                size_t residue=row&(e-1),split=e-residue;
                for(size_t col=0;col<n;col+=e) {
                    size_t diagonal=row+col;
                    int64_t *first=sums+residue*n+(diagonal%n/e)*e;
                    const uint64_t *input=values+row*n+col;
                    if(diagonal<n) for(size_t j=0;j<split;j++) first[j]+=(int64_t)input[j];
                    else for(size_t j=0;j<split;j++) first[j]-=(int64_t)input[j];
                    diagonal+=split;
                    int64_t *second=sums+residue*n+(diagonal%n/e)*e;
                    if(diagonal<n) for(size_t j=split;j<e;j++) second[j]+=(int64_t)input[j];
                    else for(size_t j=split;j<e;j++) second[j]-=(int64_t)input[j];
                }
            }
        }
        for(size_t i=0;i<batch*n*e;i++)
            weighted[i]+=(__int128)collapsed[i]*weights[d];
    }
    for(size_t i=0;i<batch*n*e;i++)collapsed[i]=(int64_t)reduce_signed128(weighted[i],q);
    for(size_t g=0;g<groups;g++) for(size_t b=0;b<batch;b++) {
        __int128 *target=acc+(g*batch+b)*e;
        for(size_t src=0;src<n;src++) {
            for(size_t slot=0;slot<e;slot++)
                target[slot]+=(__int128)source[g][src]*
                    collapsed[b*n*e+((src-slot)&(e-1))*n+(src/e)*e+slot];
        }
    }
    for(size_t i=0;i<count;i++) {
        out[i]=reduce_signed128(acc[i],q);
    }
    free(acc);free(weighted);free(collapsed);return 0;
}

int latge_sparse_integer_product(const uint64_t *offsets,const uint64_t *columns,
        const uint64_t *shifts,const int64_t *weights,size_t rows,size_t cols,
        size_t n,size_t terms,const int64_t *input,size_t batch,int64_t *output) {
    if (!offsets || !columns || !shifts || !weights || !input || !output
            || !rows || !cols || !n || !batch || offsets[0] || offsets[rows]!=terms
            || rows>SIZE_MAX/n || cols>SIZE_MAX/n || batch>SIZE_MAX/(rows*n)
            || batch>SIZE_MAX/(cols*n) || batch*rows*n>SIZE_MAX/sizeof(int64_t)) return -1;
    for (size_t r=0;r<rows;r++)
        if (offsets[r]>offsets[r+1] || offsets[r+1]>terms) return -1;
    for (size_t t=0;t<terms;t++) if (columns[t]>=cols || shifts[t]>=n) return -1;
    memset(output,0,batch*rows*n*sizeof(int64_t));
    for (size_t r=0;r<rows;r++) {
        for (size_t t=offsets[r];t<offsets[r+1];t++) {
            size_t shift=shifts[t],split=n-shift;
            int64_t weight=weights[t];
            for (size_t b=0;b<batch;b++) {
                const int64_t *source=input+(b*cols+columns[t])*n;
                int64_t *target=output+(b*rows+r)*n;
                for (size_t j=0;j<split;j++) target[j+shift]+=weight*source[j];
                for (size_t j=split;j<n;j++) target[j-split]-=weight*source[j];
            }
        }
    }
    return 0;
}
