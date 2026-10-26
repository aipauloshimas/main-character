"""The style engine: recipes, and the layers a card is drawn from.

A style is a recipe, one option for each layer (background, cutout edge, notes, arrows, one hero
element, little extras) and a palette of five colors. recipes holds the options, the palettes, the
ready-made styles and the person's saved ones. background, edges and notes draw the options, and
engine puts them together. This file imports nothing, so that the project file can use recipes
without loading the renderer. (The contrast guard, palette.py, which the project file imports, does
load background and edges, for the tone of a background; it never loads notes, engine or the card.)
"""
