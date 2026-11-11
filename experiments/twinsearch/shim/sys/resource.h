// Stand-in for the POSIX header <sys/resource.h>, which MinGW does not ship.
// twin_search.cpp includes it for one function, rss_mark(), which returns at once unless the environment variable
// TWIN_RSS_TRACE is set (never set in these runs).  Not part of the TwinSearch repository.
#pragma once
struct rusage { long ru_maxrss; };
#define RUSAGE_SELF 0
static inline int getrusage(int, struct rusage *r) { r->ru_maxrss = 0; return 0; }
