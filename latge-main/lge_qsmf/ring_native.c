#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

#define TABLE_Q UINT64_C(4294967969)
static const uint64_t table_powers[32]={
    1ULL, 375894760ULL, 738854916ULL, 1381253738ULL, 367967678ULL, 2145593509ULL, 4090952515ULL, 891285777ULL, 2505012387ULL, 607048196ULL, 3705491869ULL, 2388760146ULL, 2811517442ULL, 3645027118ULL, 1404491633ULL, 555776934ULL, 4294967968ULL, 3919073209ULL, 3556113053ULL, 2913714231ULL, 3927000291ULL, 2149374460ULL, 204015454ULL, 3403682192ULL, 1789955582ULL, 3687919773ULL, 589476100ULL, 1906207823ULL, 1483450527ULL, 649940851ULL, 2890476336ULL, 3739191035ULL
};
static const uint64_t table_untwist[16]={
    4026532471ULL, 1575876930ULL, 180654771ULL, 3530282778ULL, 361151156ULL, 387573487ULL, 3258068233ULL, 1035801480ULL, 648743220ULL, 212730137ULL, 549621962ULL, 1208077896ULL, 3735098993ULL, 2598026622ULL, 1027563560ULL, 2123990562ULL
};

static inline uint64_t table_mul(uint64_t a, uint64_t b) {
    unsigned __int128 p=(unsigned __int128)a*b;
    int64_t folded=(int64_t)(uint32_t)p
        - INT64_C(673)*(int64_t)(uint32_t)(p>>32)
        + INT64_C(452929)*(int64_t)(p>>64);
    int64_t value=folded%(int64_t)TABLE_Q;
    return (uint64_t)(value<0 ? value+(int64_t)TABLE_Q : value);
}

static inline uint64_t table_reduce(__int128 value) {
    int negative=value<0;
    unsigned __int128 v=negative ? (unsigned __int128)(-value) : (unsigned __int128)value;
    int64_t folded=(int64_t)(uint32_t)v
        - INT64_C(673)*(int64_t)(uint32_t)(v>>32)
        + INT64_C(452929)*(int64_t)(uint32_t)(v>>64)
        - INT64_C(304821217)*(int64_t)(uint32_t)(v>>96);
    int64_t reduced=folded%(int64_t)TABLE_Q;
    uint64_t canonical=(uint64_t)(reduced<0 ? reduced+(int64_t)TABLE_Q : reduced);
    return negative && canonical ? TABLE_Q-canonical : canonical;
}

static const unsigned reverse4[16]={0,8,4,12,2,10,6,14,1,9,5,13,3,11,7,15};

static void table_butterflies(uint64_t values[16], const uint64_t powers[32], int inverse) {
    for (unsigned length=2;length<=16;length<<=1) {
        unsigned half=length/2, stride=32/length;
        for (unsigned start=0;start<16;start+=length) {
            for (unsigned j=0;j<half;j++) {
                unsigned exponent=j*stride;
                uint64_t b=table_mul(values[start+j+half],powers[inverse ? (32-exponent)&31 : exponent]);
                uint64_t a=values[start+j], sum=a+b;
                values[start+j]=sum>=TABLE_Q ? sum-TABLE_Q : sum;
                values[start+j+half]=a>=b ? a-b : a+TABLE_Q-b;
            }
        }
    }
}

static void table_forward(const int64_t *input,int64_t *output,const uint64_t powers[32]) {
    for (unsigned r=0;r<8;r++) {
        uint64_t values[16];
        for (unsigned j=0;j<16;j++)
            values[reverse4[j]]=table_mul((uint64_t)input[8*j+r],powers[j]);
        table_butterflies(values,powers,0);
        for (unsigned u=0;u<16;u++)
            output[8*u+r]=values[u]>TABLE_Q/2 ? (int64_t)values[u]-(int64_t)TABLE_Q : (int64_t)values[u];
    }
}

static void table_inverse(const uint64_t *input,int64_t *output,
                          const uint64_t powers[32],const uint64_t untwist[16]) {
    for (unsigned r=0;r<8;r++) {
        uint64_t values[16];
        for (unsigned u=0;u<16;u++) values[reverse4[u]]=input[8*u+r];
        table_butterflies(values,powers,1);
        for (unsigned j=0;j<16;j++) output[8*j+r]=(int64_t)table_mul(values[j],untwist[j]);
    }
}

static int table_product(const int64_t *a,const int64_t *b,size_t rows,size_t inner,
                          size_t columns,int transpose_left,const int64_t *prepared_left,int64_t *out) {
    size_t acount=rows*inner*128,bcount=inner*columns*128;
    int64_t *owned_left=prepared_left ? NULL : malloc(acount*sizeof(int64_t));
    int64_t *tb=malloc(bcount*sizeof(int64_t));
    const int64_t *ta=prepared_left ? prepared_left : owned_left;
    if (!ta || !tb) {free(owned_left);free(tb);return -2;}
    const uint64_t *powers=table_powers,*untwist=table_untwist;
    if (!prepared_left) for (size_t i=0;i<acount;i+=128) table_forward(a+i,owned_left+i,powers);
    for (size_t i=0;i<bcount;i+=128) table_forward(b+i,tb+i,powers);
    for (size_t i=0;i<rows;i++) {
        for (size_t j=0;j<columns;j++) {
            uint64_t transformed[128];
            for (unsigned u=0;u<16;u++) {
                __int128 acc[15]={0};
                for (size_t k=0;k<inner;k++) {
                    const int64_t *aa=ta+(transpose_left ? k*rows+i : i*inner+k)*128+8*u;
                    const int64_t *bb=tb+(k*columns+j)*128+8*u;
                    for (unsigned t=0;t<8;t++) if (bb[t])
                        for (unsigned r=0;r<8;r++) acc[r+t]+=(__int128)aa[r]*bb[t];
                }
                for (unsigned r=0;r<8;r++) {
                    uint64_t value=table_reduce(acc[r]);
                    if (r<7) {
                        value+=table_mul(powers[2*u+1],table_reduce(acc[r+8]));
                        if (value>=TABLE_Q) value-=TABLE_Q;
                    }
                    transformed[8*u+r]=value;
                }
            }
            table_inverse(transformed,out+(i*columns+j)*128,powers,untwist);
        }
    }
    free(owned_left);free(tb);return 0;
}

static int ring_product(const int64_t *a, const int64_t *b,
                       size_t rows, size_t inner, size_t columns, size_t n,
                       uint64_t q, int transpose_left, const int64_t *prepared_left, int64_t *out) {
    if (!a || !b || !out || !rows || !inner || !columns || !n || n > 4096
        || q < 2 || q > INT64_MAX || (transpose_left != 0 && transpose_left != 1))
        return -1;
    unsigned __int128 product_bound = (unsigned __int128)(q/2) * (q/2);
    unsigned __int128 budget = (((unsigned __int128)1) << 126) - 1;
    if (product_bound > budget / n || product_bound * n > budget / inner)
        return -1;
    if (rows > SIZE_MAX / inner || rows * inner > SIZE_MAX / n
        || inner > SIZE_MAX / columns || inner * columns > SIZE_MAX / n
        || rows > SIZE_MAX / columns || rows * columns > SIZE_MAX / n)
        return -1;
    size_t a_count = rows * inner * n, b_count = inner * columns * n;
    if (a_count > SIZE_MAX/sizeof(int64_t) || b_count > SIZE_MAX/sizeof(int64_t)
        || rows*columns*n > SIZE_MAX/sizeof(int64_t)) return -1;
    for (size_t i=0; i<a_count; ++i) if (a[i]<0 || (uint64_t)a[i]>=q) return -1;
    size_t right_nonzero=0;
    for (size_t i=0; i<b_count; ++i) {
        if (b[i]<0 || (uint64_t)b[i]>=q) return -1;
        right_nonzero+=(b[i]!=0);
    }
    if (n==128 && q==TABLE_Q && right_nonzero>b_count/16)
        return table_product(a,b,rows,inner,columns,transpose_left,prepared_left,out);
    int64_t *centered_a = malloc(a_count * sizeof(int64_t));
    int64_t *centered_b = malloc(b_count * sizeof(int64_t));
    __int128 *acc = malloc(n * sizeof(__int128));
    if (!centered_a || !centered_b || !acc) {
        free(centered_a); free(centered_b); free(acc); return -2;
    }
    for (size_t i=0; i<a_count; ++i)
        centered_a[i] = (uint64_t)a[i]>q/2 ? a[i]-(int64_t)q : a[i];
    for (size_t i=0; i<b_count; ++i)
        centered_b[i] = (uint64_t)b[i]>q/2 ? b[i]-(int64_t)q : b[i];
    for (size_t i=0; i<rows; ++i) {
        for (size_t j=0; j<columns; ++j) {
            memset(acc, 0, n*sizeof(__int128));
            for (size_t k=0; k<inner; ++k) {
                const int64_t *aa = centered_a + (transpose_left ? k*rows+i : i*inner+k)*n;
                const int64_t *bb = centered_b + (k*columns+j)*n;
                for (size_t t=0; t<n; ++t) {
                    int64_t v = bb[t];
                    if (!v) continue;
                    for (size_t r=0; r<n-t; ++r) acc[r+t] += (__int128)aa[r]*v;
                    for (size_t r=n-t; r<n; ++r) acc[r+t-n] -= (__int128)aa[r]*v;
                }
            }
            for (size_t t=0; t<n; ++t) {
                __int128 value = acc[t] % (__int128)q;
                out[(i*columns+j)*n+t] = (int64_t)(value<0 ? value+q : value);
            }
        }
    }
    free(centered_a); free(centered_b); free(acc); return 0;
}

int latge_table_prepare(const int64_t *a,size_t polynomials,int64_t *out) {
    if (!a || !out || !polynomials || polynomials>SIZE_MAX/(128*sizeof(int64_t))) return -1;
    for (size_t i=0;i<polynomials*128;i++) if (a[i]<0 || (uint64_t)a[i]>=TABLE_Q) return -1;
    for (size_t i=0;i<polynomials*128;i+=128) table_forward(a+i,out+i,table_powers);
    return 0;
}
int latge_ring_product(const int64_t *a,const int64_t *b,size_t rows,size_t inner,
                      size_t columns,size_t n,uint64_t q,int transpose_left,int64_t *out) {
    return ring_product(a,b,rows,inner,columns,n,q,transpose_left,NULL,out);
}
int latge_ring_product_prepared(const int64_t *a,const int64_t *b,size_t rows,size_t inner,
                      size_t columns,size_t n,uint64_t q,int transpose_left,
                      const int64_t *prepared_left,int64_t *out) {
    if (!prepared_left || n!=128 || q!=TABLE_Q) return -1;
    return ring_product(a,b,rows,inner,columns,n,q,transpose_left,prepared_left,out);
}
