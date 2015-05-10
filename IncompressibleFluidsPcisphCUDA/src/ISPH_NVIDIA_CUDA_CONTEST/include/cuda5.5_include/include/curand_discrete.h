
#if !defined(CURANDDISCRETE_H_)
#define CURANDDISCRETE_H_

struct curandDistributionShift_st {
    curandDistribution_t probability;
    curandDistribution_t host_probability;
    unsigned int shift;
    unsigned int length;
    unsigned int host_gen;
};

struct curandHistogramM2_st {
    curandHistogramM2V_t V; 
    curandHistogramM2V_t host_V; 
    curandHistogramM2K_t K; 
    curandHistogramM2K_t host_K; 
    unsigned int host_gen;
};


struct curandDistributionM2Shift_st {
    curandHistogramM2_t histogram;
    curandHistogramM2_t host_histogram;
    unsigned int shift;
    unsigned int length;
    unsigned int host_gen;
};

struct curandDiscreteDistribution_st {
    curandDiscreteDistribution_t self_host_ptr;
    curandDistributionM2Shift_t M2;
    curandDistributionM2Shift_t host_M2;
    double stddev;
    double mean;
    curandMethod_t method;
    unsigned int host_gen; 
};

#endif // !defined(CURANDDISCRETE_H_)
