#define AHRI_EXPORT

#ifdef _MSC_VER
#ifdef AHRI_EXPORT
#define AHRI_API __declspec(dllexport)
#else
#define AHRI_API __declspec(dllimport)
#endif
#elif defined(__GNUC__) || defined(__GNUG__)
#define AHRI_API __attribute__((visibility("default")))
#endif
