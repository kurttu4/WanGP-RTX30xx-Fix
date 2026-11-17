# Custom user variables and functions for Wan2GP
# Kept in a separate file to prevent upstream git merges from overwriting them.

CONTINUE_NO_JOIN = [False]  # state of the Continue without joining checkbox (page only, not saved in settings)

def set_continue_no_join(value):
    CONTINUE_NO_JOIN[0] = bool(value)
