command_definitions = {
    # Canonical spelling of CPM public functions (taken from CPM 0.40.2)
    "CPMAddPackage": {
        "front_positional_arguments": "short_package_spec",
        "one_value_keywords": [
            "NAME",
            "FORCE",
            "VERSION",
            "GIT_TAG",
            "DOWNLOAD_ONLY",
            "GITHUB_REPOSITORY",
            "GITLAB_REPOSITORY",
            "BITBUCKET_REPOSITORY",
            "GIT_REPOSITORY",
            "SOURCE_DIR",
            "FIND_PACKAGE_ARGUMENTS",
            "NO_CACHE",
            "SYSTEM",
            "GIT_SHALLOW",
            "EXCLUDE_FROM_ALL",
            "SOURCE_SUBDIR",
            "CUSTOM_CACHE_KEY",
        ],
        "multi_value_keywords": ["URL", "OPTIONS", "DOWNLOAD_COMMAND", "PATCHES"],
    },
    "CPMFindPackage": {
        "one_value_keywords": ["NAME", "VERSION", "GIT_TAG", "FIND_PACKAGE_ARGUMENTS"],
    },
    "CPMRegisterPackage": {},
    "CPMGetPackageVersion": {},
}
