from dash import html

def layout() -> html.Div:
    """Layout pour le tableau de bord de qualité des données."""
    return html.Div([
        html.H2("Tableau de bord qualité", className="mt-3"),
        html.P("Cette page affichera les anomalies détectées par les contrôles de qualité. Elle est en cours de développement.")
    ])
