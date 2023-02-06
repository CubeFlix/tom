// declspec.h
// DECLSPEC definition.

#ifndef DECLSPEC_H
#define DECLSPEC_H

#ifdef _MSC_VER
#ifdef TOM_EXPORTS
#define TOM_API __declspec(dllexport)
#else
#define TOM_API __declspec(dllimport)
#endif
#else
#define TOM_API
#endif

#endif
