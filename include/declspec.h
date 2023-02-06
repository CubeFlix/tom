// declspec.h
// DECLSPEC definition.

#ifndef DECLSPEC_H
#define DECLSPEC_H

#ifdef TOM_EXPORTS
#define TOM_API __declspec(dllexport)
#else
#define TOM_API __declspec(dllimport)
#endif

#endif