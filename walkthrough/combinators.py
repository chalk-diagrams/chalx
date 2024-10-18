# %% tags=["hide_inp"]
from chalk import *
from chalk.transform import polar


# %% [markdown]
# Complex diagrams can be created by combining simpler diagrams
# through placement combinators. These place diagrams above, atop or
# besides other diagrams. Relative location is determined by the envelope
# and origins of the diagrams.




# %%
triangle(1).beside(square(1), unit_x)


# %%
triangle(1).show_beside(square(1), unit_x)

# %%
triangle(1).show_beside(triangle(1).rotate_by(1 / 6), polar(-45))


# %%
triangle(1).show_beside(triangle(1).rotate_by(1 / 6), polar(-30))


# %%
diagram = triangle(1) / square(1)
diagram


# %%
diagram.show_envelope().show_origin()





# %% [markdown]
# Example 1 - Atop at origin

# %%
diagram = square(1) + triangle(1)
diagram


# %%
diagram.show_envelope().show_origin()


# %%
s = square(1).align_r().align_b().show_origin()
t = triangle(1).align_l().align_t().show_origin()
s


# %%
t


# %%
s + t






# %%
vcat([triangle(1), square(1), triangle(1)], 0.2)





# %%
concat([triangle(1), square(1), triangle(1)])



# %%
hcat([triangle(1), square(1), triangle(1)], 0.2)



# %%
place_on_path(
    [circle(0.25) for _ in range(6)],
    Trail.regular_polygon(6, 1).to_path(),
)
