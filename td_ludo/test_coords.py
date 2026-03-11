import src.tensor_utils as tu
print("Python P0 Base:", tu.BASE_COORDS[0])
print("Python P0 Start:", tu.PATH_COORDS_P0[0])
for p in range(4):
    print(f"Python P{p} Start (Rotated):", tu.get_board_coords(p, 0))
