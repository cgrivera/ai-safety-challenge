# Changelog for TanksWorld environment

## 0.1.4

* Fixed inference mode resolution mismatch with UI which was causing UI elements to appear the wrong size - now the inference window is 1358x1131 to match the UI.
* Plugging in a game controller overrides Red Tank 1's input with the controller's input in inference mode only. Unplug the controller to give control back to the policy.

## 0.1.3

* Modified the top down obstacle view:
	* Resize to 128x128 in dimensions to prevent loss of information from low resolutions
	* Include the bounding walls of the arena
* Simple score-board added onto the cinematic view to track friendly, enemy, and collateral kills for each team.
* When an agent dies, its FPV in the game window is crossed out with a UI element.
* Bugfix: Updated code to match vector observation docs: in kill info, if the attack's victim is a neutral tank, the "team" will be "3" (was "-1" in previous versions) 
* Shell launch speed is increased from 15 m/s to 50 m/s

## 0.1.2

* Added an 80x80 top-down visual observation to describe neutral, static object placement.
* Updated docs on vector observation values.

## 0.1.1

* "Dead" agents continue to send state

## 0.1.0

* All agents send identical truth data for vector observation of length 84.

## 0.0.0

* Test build for initial integration testing