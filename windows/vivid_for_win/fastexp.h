#ifndef FASTEXP_H
#define FASTEXP_H

//Yoinked from: http://www.musicdsp.org/showone.php?id=222
inline float fastexp3(float x) { 
    return (6+x*(6+x*(3+x)))*0.16666666f; 
};

inline float fastexp4(float x) {
    return (24+x*(24+x*(12+x*(4+x))))*0.041666666f;
};

inline float fastexp5(float x) {
    return (120+x*(120+x*(60+x*(20+x*(5+x)))))*0.0083333333f;
};

inline float fastexp6(float x) {
    return 720+x*(720+x*(360+x*(120+x*(30+x*(6+x)))))*0.0013888888f;
};

inline float fastexp7(float x) {
    return (5040+x*(5040+x*(2520+x*(840+x*(210+x*(42+x*(7+x)))))))*0.00019841269f;
};

inline float fastexp8(float x) {
    return (40320+x*(40320+x*(20160+x*(6720+x*(1680+x*(336+x*(56+x*(8+x))))))))*2.4801587301e-5;
};

inline float fastexp9(float x) {
  return (362880+x*(362880+x*(181440+x*(60480+x*(15120+x*(3024+x*(504+x*(72+x*(9+x)))))))))*2.75573192e-6;
};

//max error in the 0 .. 7.5 range: ~0.45% 
inline float fastexp3_large(float const &x) { 
    if (x<2.5) return 2.7182818f * fastexp3(x-1.f); 
    else if (x<5) return 33.115452f * fastexp3(x-3.5f); 
    else return 403.42879f * fastexp3(x-6.f); 
}; 

inline float fastexp4_large(float const &x) { 
    if (x<2.5) return 2.7182818f * fastexp4(x-1.f); 
    else if (x<5) return 33.115452f * fastexp4(x-3.5f); 
    else return 403.42879f * fastexp4(x-6.f); 
}; 

inline float fastexp5_large(float const &x) { 
    if (x<2.5) return 2.7182818f * fastexp5(x-1.f); 
    else if (x<5) return 33.115452f * fastexp5(x-3.5f); 
    else return 403.42879f * fastexp5(x-6.f); 
}; 

inline float fastexp6_large(float const &x) { 
    if (x<2.5) return 2.7182818f * fastexp6(x-1.f); 
    else if (x<5) return 33.115452f * fastexp6(x-3.5f); 
    else return 403.42879f * fastexp6(x-6.f); 
}; 

inline float fastexp7_large(float const &x) { 
    if (x<2.5) return 2.7182818f * fastexp7(x-1.f); 
    else if (x<5) return 33.115452f * fastexp7(x-3.5f); 
    else return 403.42879f * fastexp7(x-6.f); 
}; 

inline float fastexp8_large(float const &x) { 
    if (x<2.5) return 2.7182818f * fastexp8(x-1.f); 
    else if (x<5) return 33.115452f * fastexp8(x-3.5f); 
    else return 403.42879f * fastexp8(x-6.f); 
}; 

inline float fastexp9_large(float const &x) { 
    if (x<2.5) return 2.7182818f * fastexp9(x-1.f); 
    else if (x<5) return 33.115452f * fastexp9(x-3.5f); 
    else return 403.42879f * fastexp9(x-6.f); 
}; 


#endif
