# Cas d'usage réel

## Centre de support multilingue en temps réel

### Contexte
Un agent support doit dialoguer avec des clients parlant plusieurs langues sans délai significatif.

### Utilisation
1. Le client parle dans son navigateur.
2. Le flux audio est envoyé au bridge `/v1/realtime`.
3. Le bridge transcrit, génère la réponse contextualisée, puis renvoie audio + texte.
4. L'agent lit/écoute la réponse et poursuit la conversation.

### Valeur opérationnelle
- Réduction du temps de traitement d'un ticket vocal.
- Uniformisation de la qualité des réponses.
- Capitalisation des conversations via transcripts/replay pour QA et formation.
