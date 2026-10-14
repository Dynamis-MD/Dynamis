cmake_minimum_required(VERSION 3.25.0)

project(
    dynamis
    VERSION 0.0.1
    DESCRIPTION "Distributed neural network potentials for molecular dynamics simulations"
    HOMEPAGE_URL "https://github.com/Dynamis-MD/Dynamis"
    LANGUAGES NONE
)

# Pure-Python package — no compiled extensions.
# Python sources are installed by scikit-build-core via pyproject.toml
# [tool.scikit-build.wheel] packages = ["src/dynamis"]
