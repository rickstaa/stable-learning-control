"""This script is used to test whether the ExperimentGrid class is working as expected.
"""

from bayesian_learning_control.control.utils.run_utils import ExperimentGrid


# Check if the ExperimentGrid class is still working as expected.
def test_eg(snapshot):
    eg = ExperimentGrid()
    eg.add("test:a", [1, 2, 3], "ta", True)
    eg.add("test:b", [1, 2, 3])
    eg.add("some", [4, 5])
    eg.add("why", [True, False])
    eg.add("huh", 5)
    eg.add("no", 6, in_name=True)

    # Ensure that the variant is equal to the snapshot.
    assert eg.variants() == snapshot
