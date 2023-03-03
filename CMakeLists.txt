cmake_minimum_required(VERSION 2.8.12)

project(tom)

file(GLOB SOURCES src/*.c)

include_directories(include)

if (MSVC)
	STRING(REGEX REPLACE "/RTC(su|[1su])" "" CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
	STRING(REGEX REPLACE "/RTC(su|[1su])" "" CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG}")
	STRING(REGEX REPLACE "/RTC(su|[1su])" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
	add_compile_options(/W4 /O2)
else()
	add_compile_options(-Wall -Wextra -Wpedantic -Ofast)
endif()

add_library(${PROJECT_NAME} SHARED ${SOURCES})

find_library(MATH_LIBRARY m)
if(MATH_LIBRARY)
    target_link_libraries(tom PUBLIC ${MATH_LIBRARY})
endif()

add_compile_definitions(TOM_EXPORTS)

install(TARGETS ${PROJECT_NAME} DESTINATION lib/${PROJECT_NAME})

file(GLOB HEADERS include/*.h)
install(FILES ${HEADERS} DESTINATION include/${PROJECT_NAME})
