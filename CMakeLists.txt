cmake_minimum_required(VERSION 2.4.0)

project(tom)

file(GLOB SOURCES src/*.c)

include_directories(include)

if (MSVC)
	add_compile_options(/W4 /O2)
else()
	add_compile_options(-Wall -Wextra -Wpedantic -Ofast)
endif()

add_library(${PROJECT_NAME} SHARED ${SOURCES})
target_link_libraries(tom m)

install(TARGETS ${PROJECT_NAME} DESTINATION lib/${PROJECT_NAME})

file(GLOB HEADERS include/*.h)
install(FILES ${HEADERS} DESTINATION include/${PROJECT_NAME})
