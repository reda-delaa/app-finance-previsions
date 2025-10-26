from dash import html

def layout() -> html.Div:
    """Layout pour la page Backtests/Evaluation montrant les métriques de performance."""
    return html.Div([
        html.H2("Backtests & Évaluation", className="mt-3"),
        html.P("Cette page présentera les métriques de performance des agents (backtests et évaluation). Elle est en cours de développement.")
    ])
