import importlib                                                               # import utilities

def abm_construct(agent_name, P, task):

    """
    This function constructs an agent from file.
    """
    return importlib.import_module(f"abm_agent_{agent_name}").Agent(P, task)   # agent construction
