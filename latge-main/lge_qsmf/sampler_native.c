#include <stdint.h>
#include <stddef.h>

typedef __uint128_t u128;
#define SCALE (UINT64_C(1) << 60)

static int table_word_less(const uint8_t *word, size_t draw_bytes,
                          uint8_t high_mask, const uint8_t *entry,
                          size_t entry_bytes) {
    for (size_t i=entry_bytes; i-- > 0;) {
        uint8_t value=i<draw_bytes ? word[i] : 0;
        if (i+1==draw_bytes) value &= high_mask;
        if (value!=entry[i]) return value<entry[i];
    }
    return 0;
}

int latge_table_gaussian_vector(const uint8_t *entropy, size_t bytes,
        size_t *position, const uint8_t *cdf, size_t entry_bytes,
        size_t draw_bytes, uint8_t high_mask, const int64_t *support,
        size_t support_count, int64_t *output, size_t count, size_t *produced) {
    if (!entry_bytes || !support_count || draw_bytes>entry_bytes
            || *position>bytes || *produced>count) return -1;
    while (*produced<count) {
        if (draw_bytes>bytes-*position) return 0;
        const uint8_t *word=entropy+*position;
        *position+=draw_bytes;
        if (!table_word_less(word,draw_bytes,high_mask,
                cdf+(support_count-1)*entry_bytes,entry_bytes)) continue;
        size_t low=0, high=support_count;
        while (low<high) {
            size_t middle=low+(high-low)/2;
            if (table_word_less(word,draw_bytes,high_mask,
                    cdf+middle*entry_bytes,entry_bytes)) high=middle;
            else low=middle+1;
        }
        output[(*produced)++]=support[low];
    }
    return 1;
}

static uint64_t ceil_div(u128 a, u128 b) {
    return (uint64_t)(a / b + (a % b != 0));
}

int latge_rho_interval(uint64_t ylo, uint64_t yhi, unsigned squares,
                       uint64_t *out) {
    if (!out || ylo > yhi || yhi > SCALE / 8 || squares > 16) return -2;
    uint64_t termlo = SCALE, termhi = SCALE;
    int64_t lower = SCALE, upper = SCALE;
    unsigned degree = 0;
    do {
        degree++;
        u128 low_product = (u128)termlo*ylo;
        u128 high_product = (u128)termhi*yhi;
        uint64_t high_word = (uint64_t)(high_product >> 60);
        termlo = (uint64_t)(low_product >> 60)/degree;
        termhi = high_word/degree
            + (high_word%degree != 0 || (high_product & (SCALE-1)) != 0);
        if (degree & 1) { lower -= termhi; upper -= termlo; }
        else { lower += termlo; upper += termhi; }
        if (degree > 128) return -3;
    } while (termhi > 1);
    lower -= termhi;
    upper += termhi;
    uint64_t lo = lower > 0 ? (uint64_t)lower : 0;
    uint64_t hi = upper < (int64_t)SCALE ? (uint64_t)upper : SCALE;
    for (unsigned i = 0; i < squares; i++) {
        lo = (uint64_t)((u128)lo * lo / SCALE);
        hi = ceil_div((u128)hi * hi, SCALE);
    }
    out[0] = lo; out[1] = hi;
    return 0;
}

int latge_rho_decide(uint64_t ylo, uint64_t yhi, unsigned squares,
                     uint64_t draw) {
    if (draw >= (UINT64_C(1) << 48) || ylo > yhi || yhi > SCALE/8
            || squares > 16) return -2;
    unsigned exponent = (unsigned)(((u128)ylo << squares) >> 60);
    if (exponent && draw && (exponent >= 48
            || draw >= (UINT64_C(1) << (48-exponent)))) return 0;
    uint64_t termlo=SCALE, termhi=SCALE;
    int64_t lower=SCALE, upper=SCALE, odd_lower=0;
    for (unsigned degree=1; degree<=4; degree++) {
        u128 low_product=(u128)termlo*ylo;
        u128 high_product=(u128)termhi*yhi;
        uint64_t high_word=(uint64_t)(high_product >> 60);
        termlo=(uint64_t)(low_product >> 60)/degree;
        termhi=high_word/degree
            +(high_word%degree != 0 || (high_product & (SCALE-1)) != 0);
        if (degree&1) {lower-=termhi; upper-=termlo;}
        else {lower+=termlo; upper+=termhi;}
        if (degree==3) odd_lower=lower;
    }
    uint64_t lo=odd_lower>0 ? (uint64_t)odd_lower : 0;
    uint64_t hi=upper<(int64_t)SCALE ? (uint64_t)upper : SCALE;
    for (unsigned i=0; i<squares; i++) {
        lo=(uint64_t)((u128)lo*lo >> 60);
        u128 square=(u128)hi*hi;
        hi=(uint64_t)(square >> 60)+((square & (SCALE-1)) != 0);
    }
    if ((draw+1) << 12 <= lo) return 1;
    if (draw << 12 >= hi) return 0;
    uint64_t interval[2];
    int status = latge_rho_interval(ylo, yhi, squares, interval);
    if (status) return status;
    if ((draw + 1) << 12 <= interval[0]) return 1;
    if (draw << 12 >= interval[1]) return 0;
    return -1;
}

int latge_gaussian_vector(const unsigned char *bytes, size_t length,
                         size_t *position, int64_t start, uint64_t width,
                         uint64_t variance, uint64_t pi_lo, uint64_t pi_hi,
                         int64_t *output, size_t count, size_t *produced,
                         int *pending, int64_t *candidate, uint64_t *draw_out) {
    if (!bytes || !position || !output || !produced || !pending ||
        !candidate || !draw_out || *position > length || *produced > count ||
        (*pending != 0 && *pending != 1) || !width || width > (UINT64_C(1)<<32) ||
        start < INT32_MIN || start > INT32_MAX ||
        start+(int64_t)width-1 > INT32_MAX || !variance || variance > (UINT64_C(1)<<56) ||
        pi_lo < 3*SCALE || pi_hi >= 4*SCALE || pi_lo > pi_hi) return -2;
    if (*pending && (*candidate < start || *candidate >= start+(int64_t)width)) return -2;
    unsigned bits = 0;
    for (uint64_t top = width-1; top; top >>= 1) bits++;
    unsigned word_bytes = (bits+7)/8;
    uint64_t mask = (UINT64_C(1)<<bits)-1;
    while (*produced < count) {
        if (!*pending) {
            uint64_t uniform;
            do {
                if (length-*position < word_bytes) return 0;
                uniform = 0;
                for (unsigned j=0; j<word_bytes; j++)
                    uniform |= (uint64_t)bytes[(*position)++] << (8*j);
                uniform &= mask;
            } while (uniform >= width);
            *candidate = start+(int64_t)uniform;
            if (!*candidate) {
                output[(*produced)++] = 0;
                continue;
            }
            *pending = 1;
        }
        if (length-*position < 6) return 0;
        uint64_t draw = 0;
        for (unsigned j=0; j<6; j++)
            draw |= (uint64_t)bytes[(*position)++] << (8*j);
        *pending = 0;
        int64_t value = *candidate;
        uint64_t magnitude = value < 0 ? (uint64_t)(-value) : (uint64_t)value;
        u128 square = (u128)magnitude*magnitude;
        uint64_t integral = (uint64_t)square/variance;
        if (draw && integral && (integral >= 16
                || draw >= (UINT64_C(1) << (48-3*integral)))) continue;
        u128 low = square*pi_lo, high = square*pi_hi;
        int decision;
        if (low >= ((u128)variance << 67)) {
            decision = draw ? 0 : -1;
        } else {
            unsigned squares = 0;
            u128 threshold = (u128)variance << 57;
            while (high > threshold) {
                threshold <<= 1;
                if (++squares > 16) return -3;
            }
            u128 divisor = (u128)variance << squares;
            uint64_t ylo = (uint64_t)(low/divisor);
            uint64_t yhi = ceil_div(high, divisor);
            decision = latge_rho_decide(ylo, yhi, squares, draw);
        }
        if (decision < -1) return decision;
        if (decision == -1) {
            *draw_out = draw;
            return 2;
        }
        if (decision) output[(*produced)++] = value;
    }
    return 1;
}

#define GPV_MAX_DIGITS 8
typedef struct {
    uint64_t q;
    uint32_t digits;
    int64_t columns[GPV_MAX_DIGITS*GPV_MAX_DIGITS];
    int64_t weights[GPV_MAX_DIGITS*GPV_MAX_DIGITS];
    uint64_t denominators[GPV_MAX_DIGITS], radii[GPV_MAX_DIGITS];
    uint64_t rho_low[GPV_MAX_DIGITS], rho_high[GPV_MAX_DIGITS];
    uint64_t pi_lo, pi_hi;
} GPVTemplate;

typedef struct {
    size_t produced;
    int level, pending;
    int64_t center[GPV_MAX_DIGITS], candidate;
    uint64_t draw;
} GPVState;

static int ratio_decision(u128 numerator, u128 denominator,
                          uint64_t pi_lo, uint64_t pi_hi, uint64_t draw) {
    u128 quotient=numerator/denominator, remainder=numerator%denominator;
    if (quotient>=32) return draw ? 0 : -1;
    if (draw && quotient && (quotient>=16
            || draw >= (UINT64_C(1) << (48-3*(unsigned)quotient)))) return 0;
    unsigned room = denominator>>64
        ? (unsigned)__builtin_clzll((uint64_t)(denominator>>64))
        : 64+(unsigned)__builtin_clzll((uint64_t)denominator);
    unsigned remaining=60;
    while (remaining) {
        unsigned shift=room<remaining ? room : remaining;
        remainder<<=shift;
        quotient=(quotient<<shift)+remainder/denominator;
        remainder%=denominator;
        remaining-=shift;
    }
    u128 low=quotient*pi_lo, high=(quotient+(remainder!=0))*pi_hi;
    unsigned squares=0;
    while (high>((u128)1<<(117+squares))) squares++;
    unsigned shift=60+squares;
    uint64_t ylo=(uint64_t)(low>>shift);
    uint64_t yhi=(uint64_t)((high>>shift)+((high&(((u128)1<<shift)-1))!=0));
    return latge_rho_decide(ylo,yhi,squares,draw);
}

typedef struct { uint64_t limb[4]; } U256;
typedef struct {
    int64_t start;
    uint64_t width, numerator[2], denominator[2], pi_lo[3], pi_hi[3];
} RationalGaussian;

static unsigned bits128(u128 x) {
    uint64_t high=(uint64_t)(x>>64);
    return high ? 128-(unsigned)__builtin_clzll(high)
                : x ? 64-(unsigned)__builtin_clzll((uint64_t)x) : 0;
}
static unsigned bits256(U256 x) {
    for (int i=3;i>=0;i--) if (x.limb[i])
        return (unsigned)(64*i)+64-(unsigned)__builtin_clzll(x.limb[i]);
    return 0;
}
static U256 multiply_pi(u128 x,const uint64_t *pi) {
    uint64_t out[5]={0},a[2]={(uint64_t)x,(uint64_t)(x>>64)};
    for (unsigned i=0;i<2;i++) {
        u128 carry=0;
        for (unsigned j=0;j<3;j++) {
            u128 value=(u128)a[i]*pi[j]+out[i+j]+carry;
            out[i+j]=(uint64_t)value;carry=value>>64;
        }
        out[i+3]=(uint64_t)carry;
    }
    U256 result={{out[0],out[1],out[2],out[3]}};
    return result;
}
static U256 shifted128(u128 x,unsigned shift) {
    U256 result={{0,0,0,0}};
    unsigned words=shift/64,bits=shift%64;
    uint64_t a[2]={(uint64_t)x,(uint64_t)(x>>64)};
    for (unsigned i=0;i<2;i++) {
        if (words+i<4) result.limb[words+i]|=a[i]<<bits;
        if (bits && words+i+1<4) result.limb[words+i+1]|=a[i]>>(64-bits);
    }
    return result;
}
static int compare256(U256 a,U256 b) {
    for (int i=3;i>=0;i--) if (a.limb[i]!=b.limb[i])
        return a.limb[i]>b.limb[i] ? 1 : -1;
    return 0;
}
static int compare_shift(U256 a,u128 b,unsigned shift) {
    if (bits128(b)+shift>256) return -1;
    return compare256(a,shifted128(b,shift));
}
static U256 subtract256(U256 a,U256 b) {
    uint64_t borrow=0;
    for (unsigned i=0;i<4;i++) {
        uint64_t difference=a.limb[i]-b.limb[i];
        uint64_t next=(a.limb[i]<b.limb[i]) | (difference<borrow);
        a.limb[i]=difference-borrow;borrow=next;
    }
    return a;
}
static uint64_t divide_shift(U256 a,u128 b,unsigned shift,int *remainder) {
    int top=(int)bits256(a)-(int)bits128(b)-(int)shift;
    uint64_t quotient=0;
    for (int i=top;i>=0;i--) {
        U256 divisor=shifted128(b,shift+(unsigned)i);
        if (compare256(a,divisor)>=0) {
            a=subtract256(a,divisor);quotient|=UINT64_C(1)<<i;
        }
    }
    *remainder=bits256(a)!=0;
    return quotient;
}
static int rational_decision(u128 numerator,u128 denominator,
                             const RationalGaussian *g,uint64_t draw) {
    U256 low=multiply_pi(numerator,g->pi_lo),high=multiply_pi(numerator,g->pi_hi);
    if (compare_shift(low,denominator,135)>=0) return draw ? 0 : -1;
    int initial=(int)bits256(high)-(int)bits128(denominator)-128+4;
    unsigned squares=initial>0 ? (unsigned)initial : 0;
    while (compare_shift(high,denominator,125+squares)>0) squares++;
    if (squares>16) return -3;
    int remainder;
    uint64_t ylo=divide_shift(low,denominator,68+squares,&remainder);
    uint64_t yhi=divide_shift(high,denominator,68+squares,&remainder);
    yhi+=remainder;
    return latge_rho_decide(ylo,yhi,squares,draw);
}

int latge_rational_gaussian_vector(const unsigned char *bytes,size_t length,
        size_t *position,const RationalGaussian *g,int64_t *output,size_t count,
        size_t *produced,int *pending,int64_t *candidate,uint64_t *draw_out) {
    if (!bytes || !position || !g || !output || !produced || !pending || !candidate
            || !draw_out || *position>length || *produced>count
            || (*pending!=0 && *pending!=1) || !g->width || g->width>(UINT64_C(1)<<32)
            || g->start<INT32_MIN || g->start>INT32_MAX
            || g->start+(int64_t)g->width-1>INT32_MAX
            || g->pi_lo[2]!=3 || g->pi_hi[2]!=3) return -2;
    u128 variance_num=((u128)g->numerator[1]<<64)|g->numerator[0];
    u128 variance_den=((u128)g->denominator[1]<<64)|g->denominator[0];
    int64_t end=g->start+(int64_t)g->width-1;
    uint64_t left=g->start<0 ? (uint64_t)(-g->start) : (uint64_t)g->start;
    uint64_t right=end<0 ? (uint64_t)(-end) : (uint64_t)end;
    uint64_t magnitude=left>right ? left : right;
    u128 max_square=(u128)magnitude*magnitude;
    if (!variance_num || variance_num>=((u128)1<<120) || !variance_den
            || (max_square && variance_den>(((u128)1<<126)-1)/max_square)) return -2;
    if (*pending && (*candidate<g->start || *candidate> end)) return -2;
    unsigned bits=0;
    for (uint64_t top=g->width-1;top;top>>=1) bits++;
    unsigned word_bytes=(bits+7)/8;
    uint64_t mask=(UINT64_C(1)<<bits)-1;
    while (*produced<count) {
        if (!*pending) {
            uint64_t uniform;
            do {
                if (length-*position<word_bytes) return 0;
                uniform=0;
                for (unsigned j=0;j<word_bytes;j++) uniform|=(uint64_t)bytes[(*position)++]<<(8*j);
                uniform&=mask;
            } while (uniform>=g->width);
            *candidate=g->start+(int64_t)uniform;
            if (!*candidate) {output[(*produced)++]=0;continue;}
            *pending=1;
        }
        if (length-*position<6) return 0;
        uint64_t draw=0;
        for (unsigned j=0;j<6;j++) draw|=(uint64_t)bytes[(*position)++]<<(8*j);
        *pending=0;
        uint64_t absolute=*candidate<0 ? (uint64_t)(-*candidate) : (uint64_t)*candidate;
        u128 numerator=(u128)absolute*absolute*variance_den;
        int decision=rational_decision(numerator,variance_num,g,draw);
        if (decision < -1) return decision;
        if (decision == -1) {*draw_out=draw;return 2;}
        if (decision) output[(*produced)++]=*candidate;
    }
    return 1;
}

int latge_gpv_vector(const unsigned char *bytes,size_t length,size_t *position,
                     const GPVTemplate *geometry,const int64_t *targets,
                     size_t count,int64_t *output,GPVState *state) {
    if (!bytes || !position || !geometry || !targets || !output || !state
            || *position>length || state->produced>count
            || !geometry->digits || geometry->digits>GPV_MAX_DIGITS
            || state->level<0 || state->level>(int)geometry->digits
            || (state->pending!=0 && state->pending!=1 && state->pending!=3)
            || geometry->q<2 || geometry->q>INT64_MAX
            || geometry->pi_lo<3*SCALE || geometry->pi_hi>=4*SCALE
            || geometry->pi_lo>geometry->pi_hi) return -2;
    unsigned digits=geometry->digits;
    while (state->produced<count) {
        if (state->level==(int)digits) {
            if (targets[state->produced]<0 || (uint64_t)targets[state->produced]>=geometry->q)
                return -2;
            for (unsigned j=0;j<digits;j++) state->center[j]=0;
            state->center[0]=-targets[state->produced];
            state->level=(int)digits-1;
        }
        unsigned level=(unsigned)state->level;
        if (state->pending==3) {
            for (unsigned j=0;j<digits;j++)
                state->center[j]-=state->candidate*geometry->columns[level*GPV_MAX_DIGITS+j];
            state->pending=0;
            if (--state->level<0) {
                for (unsigned j=0;j<digits;j++)
                    output[state->produced*digits+j]=-state->center[j];
                state->produced++;
                state->level=(int)digits;
            }
            continue;
        }
        uint64_t denominator=geometry->denominators[level];
        __int128 numerator=0;
        for (unsigned j=0;j<digits;j++)
            numerator+=(__int128)state->center[j]*geometry->weights[level*GPV_MAX_DIGITS+j];
        __int128 floor=numerator/denominator, remainder=numerator%denominator;
        if (remainder<0) {floor--;remainder+=denominator;}
        if (!state->pending) {
            uint64_t width=2*geometry->radii[level]+2,uniform;
            unsigned bits=0;
            for (uint64_t top=width-1;top;top>>=1) bits++;
            unsigned word_bytes=(bits+7)/8;
            uint64_t mask=(UINT64_C(1)<<bits)-1;
            do {
                if (length-*position<word_bytes) return 0;
                uniform=0;
                for (unsigned j=0;j<word_bytes;j++)
                    uniform|=(uint64_t)bytes[(*position)++]<<(8*j);
                uniform&=mask;
            } while (uniform>=width);
            state->candidate=(int64_t)floor-(int64_t)geometry->radii[level]+(int64_t)uniform;
            state->pending=1;
        }
        __int128 displacement=(__int128)state->candidate*denominator-numerator;
        u128 magnitude=displacement<0 ? (u128)(-displacement) : (u128)displacement;
        uint64_t nearest=(uint64_t)remainder;
        if (nearest>denominator-nearest) nearest=denominator-nearest;
        u128 factor=magnitude*magnitude-(u128)nearest*nearest;
        if (!factor) {state->pending=3;continue;}
        if (length-*position<6) return 0;
        state->draw=0;
        for (unsigned j=0;j<6;j++) state->draw|=(uint64_t)bytes[(*position)++]<<(8*j);
        state->pending=0;
        u128 rho_denominator=((u128)geometry->rho_high[level]<<64)|geometry->rho_low[level];
        int decision=ratio_decision(factor,rho_denominator,
                                    geometry->pi_lo,geometry->pi_hi,state->draw);
        if (decision < -1) return decision;
        if (decision==-1) {state->pending=2;return 2;}
        if (decision) state->pending=3;
    }
    return 1;
}

#define GAUSSIAN_BANDS 256
typedef struct {
    uint64_t variance, pi_lo, pi_hi, bands;
    int64_t start[GAUSSIAN_BANDS];
    uint64_t width[GAUSSIAN_BANDS], lower[GAUSSIAN_BANDS], upper[GAUSSIAN_BANDS];
    uint64_t cumulative_lo[GAUSSIAN_BANDS], cumulative_hi[GAUSSIAN_BANDS];
} GaussianEnvelope;
typedef struct {int stage,band;int64_t candidate;uint64_t draw;} EnvelopeState;
static int envelope_uniform(const unsigned char *bytes,size_t length,size_t *position,
                            u128 bound,u128 *value){
    if(!bound)return -1;
    unsigned bits=0;for(u128 x=bound-1;x;x>>=1)bits++;
    unsigned width=(bits+7)/8;u128 mask=bits?(((u128)1<<bits)-1):0;
    do {
        if(length-*position<width)return 0;
        u128 v=0;for(unsigned j=0;j<width;j++)v|=(u128)bytes[(*position)++]<<(8*j);
        *value=v&mask;
    } while(*value>=bound);
    return 1;
}
int latge_envelope_gaussian(const unsigned char *bytes,size_t length,size_t *position,
    const GaussianEnvelope *g,int64_t *output,size_t count,size_t *produced,EnvelopeState *state){
    if(!g||!g->bands||g->bands>GAUSSIAN_BANDS||*position>length||*produced>count)return -1;
    u128 total=((u128)g->cumulative_hi[g->bands-1]<<64)|g->cumulative_lo[g->bands-1];
    while(*produced<count){
        u128 value;int ready;
        if(state->stage==0){
            ready=envelope_uniform(bytes,length,position,total,&value);if(ready!=1)return ready;
            unsigned lo=0,hi=(unsigned)g->bands;
            while(lo<hi){unsigned mid=(lo+hi)/2;
                u128 end=((u128)g->cumulative_hi[mid]<<64)|g->cumulative_lo[mid];
                if(value<end)hi=mid;else lo=mid+1;
            }
            state->band=(int)lo;state->stage=1;
        }
        unsigned band=(unsigned)state->band;
        if(state->stage==1){
            ready=envelope_uniform(bytes,length,position,g->width[band],&value);if(ready!=1)return ready;
            state->candidate=g->start[band]+(int64_t)value;state->stage=2;
        }
        ready=envelope_uniform(bytes,length,position,g->upper[band],&value);if(ready!=1)return ready;
        uint64_t draw=(uint64_t)value;state->stage=0;
        if(draw<g->lower[band]){output[(*produced)++]=state->candidate;continue;}
        uint64_t magnitude=state->candidate<0?(uint64_t)(-state->candidate):(uint64_t)state->candidate;
        u128 square=(u128)magnitude*magnitude;
        u128 low=square*g->pi_lo,high=square*g->pi_hi;
        unsigned squares=0;u128 threshold=(u128)g->variance<<57;
        while(high>threshold){threshold<<=1;if(++squares>16)return -1;}
        u128 divisor=(u128)g->variance<<squares;
        int decision=latge_rho_decide((uint64_t)(low/divisor),ceil_div(high,divisor),squares,draw);
        if(decision==-1){state->draw=draw;return 2;}
        if(decision<0)return -1;
        if(decision)output[(*produced)++]=state->candidate;
    }
    return 1;
}
