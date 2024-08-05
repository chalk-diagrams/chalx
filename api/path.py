from jaxtyping import install_import_hook
with install_import_hook("chalk", "typeguard.typechecked"):
    import chalk 
from chalk import *


Path.from_list_of_tuples([(1,2), (2, 1)]).stroke()