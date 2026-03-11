import td_ludo.src.tensor_utils as tu
print("Python P0 Base:", tu.BASE_COORDS[0])
print("Python P0 Start:", tu.PATH_COORDS_P0[0])
print("Python P2 Base:", tu.BASE_COORDS[2])
print("Python P2 Start:", tu.PATH_COORDS_P0[0]) # (Wait, P2 start is rotated)
print("Python P2 Start (Rotated):", tu.get_board_coords(2, 0))
