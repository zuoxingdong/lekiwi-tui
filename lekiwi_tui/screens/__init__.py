"""screens — one ScreenState per LeKiwi action.

Each module exposes a ``ScreenState`` subclass constructed as ``ScreenClass(app, ctx)``
(see lekiwi_tui.dispatch for the authoring convention). The dispatcher lazily imports
the one it needs, so importing this package pulls in no screen by itself.
"""
