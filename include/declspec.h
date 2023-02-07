// declspec.h
// DECLSPEC definition.

#ifndef DECLSPEC_H
#define DECLSPEC_H

#if defined(_WIN32)
#ifdef TOM_EXPORTS
#define TOM_API __declspec(dllexport)
#else
#define TOM_API __declspec(dllimport)
#endif
#else
#define TOM_API __attribute__((visibility("default")))
#endif

#endif
