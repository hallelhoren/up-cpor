// cpp/ff_shims.cpp
extern "C" {
    // Legacy FF symbols required by ff_api.c that we stub out
    void make_search_space(void) {}
    void ff_reset_search_state(void) {}
    void ff_clear_hash_table(void) {}
}