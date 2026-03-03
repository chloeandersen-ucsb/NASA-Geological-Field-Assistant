"""
Rock Volume & Mass Estimator v0
A simple, deterministic pipeline to estimate object dimensions, volume, and mass
from a single image using a known reference object.

Constraints:
- No machine learning, no cloud services, no automation beyond basic math
- Manual pixel measurements from image required
- Proof-of-concept for field use
"""

import math
from dataclasses import dataclass
from typing import Tuple


@dataclass
class ReferenceObject:
    """Calibration object with known real-world size."""
    name: str
    real_diameter_cm: float
    pixel_diameter: float
    
    def pixel_to_cm_ratio(self) -> float:
        """Calculate conversion factor: cm per pixel."""
        return self.real_diameter_cm / self.pixel_diameter


@dataclass
class RockMeasurement:
    """Pixel measurements of the rock from image."""
    length_pixels: float
    width_pixels: float
    height_pixels: float
    
    def to_cm(self, scale_factor: float) -> Tuple[float, float, float]:
        """Convert pixel measurements to centimeters."""
        return (
            self.length_pixels * scale_factor,
            self.width_pixels * scale_factor,
            self.height_pixels * scale_factor
        )


@dataclass
class MaterialProperties:
    """Material density properties."""
    name: str
    min_density_g_cm3: float
    max_density_g_cm3: float


class RockVolumeEstimator:
    """Estimates rock volume and mass from image measurements."""
    
    def __init__(
        self,
        reference: ReferenceObject,
        material: MaterialProperties
    ):
        self.reference = reference
        self.material = material
        self.scale_factor = reference.pixel_to_cm_ratio()
    
    def ellipsoid_volume(
        self,
        length_cm: float,
        width_cm: float,
        height_cm: float
    ) -> float:
        """
        Calculate ellipsoid volume.
        
        Formula: V = (4/3) * π * (a) * (b) * (c)
        where a, b, c are semi-axes (half of L, W, H)
        
        Args:
            length_cm: Length dimension in cm
            width_cm: Width dimension in cm
            height_cm: Height dimension in cm
            
        Returns:
            Volume in cm³
        """
        semi_a = length_cm / 2
        semi_b = width_cm / 2
        semi_c = height_cm / 2
        
        volume = (4 / 3) * math.pi * semi_a * semi_b * semi_c
        return volume
    
    def estimate(self, rock: RockMeasurement) -> dict:
        """
        Estimate rock dimensions, volume, and mass range.
        
        Args:
            rock: RockMeasurement with pixel measurements
            
        Returns:
            Dictionary with dimensions, volume, and mass estimates
        """
        # Convert pixels to centimeters
        length_cm, width_cm, height_cm = rock.to_cm(self.scale_factor)
        
        # Calculate volume using ellipsoid model
        volume_cm3 = self.ellipsoid_volume(length_cm, width_cm, height_cm)
        
        # Calculate mass range based on density bounds
        mass_min_g = volume_cm3 * self.material.min_density_g_cm3
        mass_max_g = volume_cm3 * self.material.max_density_g_cm3
        
        # Convert to kilograms
        mass_min_kg = mass_min_g / 1000
        mass_max_kg = mass_max_g / 1000
        
        return {
            "length_cm": length_cm,
            "width_cm": width_cm,
            "height_cm": height_cm,
            "volume_cm3": volume_cm3,
            "mass_min_kg": mass_min_kg,
            "mass_max_kg": mass_max_kg,
            "mass_range_kg": (mass_min_kg, mass_max_kg)
        }
    
    def print_summary(self, rock: RockMeasurement) -> None:
        """Print a human-readable summary of estimates."""
        results = self.estimate(rock)
        
        print("\n" + "="*70)
        print("ROCK VOLUME & MASS ESTIMATION - v0")
        print("="*70)
        
        print("\n📏 REFERENCE OBJECT")
        print(f"   {self.reference.name}: {self.reference.real_diameter_cm} cm diameter")
        print(f"   Pixel measurement: {self.reference.pixel_diameter} px")
        print(f"   Scale factor: {self.scale_factor:.6f} cm/px")
        
        print("\n📐 MEASURED DIMENSIONS (pixels)")
        print(f"   Length: {rock.length_pixels:.1f} px")
        print(f"   Width:  {rock.width_pixels:.1f} px")
        print(f"   Height: {rock.height_pixels:.1f} px")
        
        print("\n📏 ESTIMATED DIMENSIONS (cm)")
        print(f"   Length: {results['length_cm']:.2f} cm")
        print(f"   Width:  {results['width_cm']:.2f} cm")
        print(f"   Height: {results['height_cm']:.2f} cm")
        
        print("\n📊 VOLUME CALCULATION")
        print(f"   Model: Ellipsoid")
        print(f"   Formula: V = (4/3)π * (L/2) * (W/2) * (H/2)")
        print(f"   Volume: {results['volume_cm3']:.2f} cm³")
        
        print("\n🪨 MATERIAL PROPERTIES")
        print(f"   Material: {self.material.name}")
        print(f"   Density range: {self.material.min_density_g_cm3}-{self.material.max_density_g_cm3} g/cm³")
        
        print("\n⚖️ ESTIMATED MASS")
        print(f"   Min (lighter rock): {results['mass_min_kg']:.3f} kg")
        print(f"   Max (denser rock):  {results['mass_max_kg']:.3f} kg")
        print(f"   Range: {results['mass_min_kg']:.3f} - {results['mass_max_kg']:.3f} kg")
        
        print("\n⚠️ ASSUMPTIONS & LIMITATIONS")
        print("   • Object approximated as an ellipsoid")
        print("   • Uniform density assumed")
        print("   • Density based on typical rock type")
        print("   • Perspective distortion not corrected")
        print("   • Pixel measurements are manual and may have error")
        print("   • This is a proof-of-concept estimate, not precise measurement")
        
        print("\n" + "="*70 + "\n")


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Define reference object (US Quarter)
    reference = ReferenceObject(
        name="US Quarter",
        real_diameter_cm=2.426,
        pixel_diameter=120.0  # pixels in image
    )
    
    # Define material (typical igneous rock)
    rock_material = MaterialProperties(
        name="Igneous Rock (Granite-like)",
        min_density_g_cm3=2.5,  # lighter rocks
        max_density_g_cm3=3.0   # denser rocks
    )
    
    # Create estimator
    estimator = RockVolumeEstimator(reference, rock_material)
    
    # Example measurement 1: Medium-sized rock
    print("\n--- EXAMPLE 1: Medium-sized rock ---")
    rock_1 = RockMeasurement(
        length_pixels=320.0,
        width_pixels=280.0,
        height_pixels=250.0
    )
    estimator.print_summary(rock_1)
    
    # Example measurement 2: Smaller rock
    print("\n--- EXAMPLE 2: Smaller rock ---")
    rock_2 = RockMeasurement(
        length_pixels=200.0,
        width_pixels=180.0,
        height_pixels=160.0
    )
    estimator.print_summary(rock_2)
    
    # Example measurement 3: Large, roughly spherical rock
    print("\n--- EXAMPLE 3: Large, roughly spherical rock ---")
    rock_3 = RockMeasurement(
        length_pixels=400.0,
        width_pixels=400.0,
        height_pixels=400.0
    )
    estimator.print_summary(rock_3)
