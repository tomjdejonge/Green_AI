import anvil.server


@anvil.server.callable
def greet():
    return 'Hello'
