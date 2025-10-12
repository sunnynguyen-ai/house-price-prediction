"""
Input validation for house price prediction
Validates features against defined ranges and data types
"""

from typing import Dict, Tuple, List, Any
import numpy as np


class InputValidator:
    """Validates input features for house price prediction"""
    
    def __init__(self, feature_ranges: Dict[str, Tuple[float, float]]):
        """
        Initialize validator with feature ranges
        
        Args:
            feature_ranges: Dictionary mapping feature names to (min, max) tuples
                           Example: {'MedInc': (0.5, 15.0), 'HouseAge': (1, 52)}
        """
        self.feature_ranges = feature_ranges
        self.required_features = set(feature_ranges.keys())
    
    def validate_features(self, features: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate all input features
        
        Args:
            features: Dictionary of feature names to values
            
        Returns:
            Tuple of (is_valid: bool, errors: List[str])
            - is_valid: True if all validations pass, False otherwise
            - errors: List of error messages describing validation failures
            
        Example:
            >>> validator = InputValidator({'MedInc': (0.5, 15.0)})
            >>> is_valid, errors = validator.validate_features({'MedInc': 3.8})
            >>> print(is_valid)  # True
            >>> print(errors)    # []
        """
        errors = []
        
        # Check all required features are present
        provided_features = set(features.keys())
        missing = self.required_features - provided_features
        
        if missing:
            missing_list = sorted(list(missing))
            errors.append(f"Missing required features: {', '.join(missing_list)}")
            return False, errors
        
        # Check for extra/unknown features
        extra = provided_features - self.required_features
        if extra:
            extra_list = sorted(list(extra))
            errors.append(f"Unknown features provided: {', '.join(extra_list)}")
        
        # Validate each required feature
        for feature_name in self.required_features:
            value = features.get(feature_name)
            error = self._validate_single_feature(feature_name, value)
            if error:
                errors.append(error)
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    def _validate_single_feature(self, feature_name: str, value: Any) -> str:
        """
        Validate a single feature
        
        Args:
            feature_name: Name of the feature
            value: Value to validate
            
        Returns:
            Error message if validation fails, empty string if valid
        """
        # Check if feature is recognized
        if feature_name not in self.feature_ranges:
            return f"Unknown feature: {feature_name}"
        
        # Check data type
        if not isinstance(value, (int, float)):
            return f"{feature_name} must be a number, got {type(value).__name__}"
        
        # Check for NaN or infinity
        if np.isnan(value):
            return f"{feature_name} cannot be NaN (Not a Number)"
        
        if np.isinf(value):
            return f"{feature_name} cannot be infinite"
        
        # Check range
        min_val, max_val = self.feature_ranges[feature_name]
        if not (min_val <= value <= max_val):
            return (
                f"{feature_name} must be between {min_val} and {max_val}, "
                f"got {value}"
            )
        
        # All validations passed
        return ""
    
    def validate_single_value(self, feature_name: str, value: Any) -> Tuple[bool, str]:
        """
        Validate a single feature value (convenience method)
        
        Args:
            feature_name: Name of the feature
            value: Value to validate
            
        Returns:
            Tuple of (is_valid: bool, error_message: str)
            
        Example:
            >>> validator = InputValidator({'MedInc': (0.5, 15.0)})
            >>> is_valid, error = validator.validate_single_value('MedInc', 3.8)
            >>> print(is_valid)  # True
        """
        error = self._validate_single_feature(feature_name, value)
        return (error == "", error)
    
    def get_feature_info(self) -> Dict[str, Dict[str, float]]:
        """
        Get information about all features including their valid ranges
        
        Returns:
            Dictionary with feature info including min, max, and range
            
        Example:
            >>> validator = InputValidator({'MedInc': (0.5, 15.0)})
            >>> info = validator.get_feature_info()
            >>> print(info['MedInc'])
            {'min': 0.5, 'max': 15.0, 'range': 14.5}
        """
        return {
            feature: {
                'min': min_val,
                'max': max_val,
                'range': max_val - min_val,
                'description': FEATURE_DESCRIPTIONS.get(feature, 'No description available')
            }
            for feature, (min_val, max_val) in self.feature_ranges.items()
        }
    
    def get_sample_valid_input(self) -> Dict[str, float]:
        """
        Generate a sample valid input using midpoint values
        
        Returns:
            Dictionary with sample valid values for all features
            
        Example:
            >>> validator = InputValidator({'MedInc': (0.5, 15.0), 'HouseAge': (1, 52)})
            >>> sample = validator.get_sample_valid_input()
            >>> print(sample)
            {'MedInc': 7.75, 'HouseAge': 26.5}
        """
        return {
            feature: (min_val + max_val) / 2
            for feature, (min_val, max_val) in self.feature_ranges.items()
        }
    
    def __repr__(self) -> str:
        """String representation of validator"""
        return f"InputValidator(features={len(self.feature_ranges)})"


# Feature descriptions for user-facing documentation
FEATURE_DESCRIPTIONS = {
    'MedInc': 'Median household income in the block group (in $10,000s). Higher income typically correlates with higher property values.',
    'HouseAge': 'Median age of houses in the block group (in years). Newer homes may command higher prices, though vintage homes in desirable areas can also be valuable.',
    'AveRooms': 'Average number of rooms per household. More rooms generally indicate larger, more valuable properties.',
    'AveBedrms': 'Average number of bedrooms per household. Typically ranges from 1-4 bedrooms for residential properties.',
    'Population': 'Total population in the block group. Indicates neighborhood density and development.',
    'AveOccup': 'Average household occupancy (persons per household). Lower values may indicate more spacious housing.',
    'Latitude': 'Geographic latitude of the block group. Affects climate, proximity to coast, and regional pricing.',
    'Longitude': 'Geographic longitude of the block group. Combined with latitude, determines specific location within California.'
}


# Feature units for display
FEATURE_UNITS = {
    'MedInc': '$10,000s',
    'HouseAge': 'years',
    'AveRooms': 'rooms/household',
    'AveBedrms': 'bedrooms/household',
    'Population': 'people',
    'AveOccup': 'people/household',
    'Latitude': 'degrees',
    'Longitude': 'degrees'
}


# Typical/reasonable values for reference
TYPICAL_VALUES = {
    'MedInc': 3.8,      # $38,000 median income
    'HouseAge': 28,      # 28 years old
    'AveRooms': 5.4,     # 5.4 rooms per household
    'AveBedrms': 1.1,    # 1.1 bedrooms per household
    'Population': 3000,  # 3000 people in block
    'AveOccup': 3.2,     # 3.2 people per household
    'Latitude': 34.2,    # Los Angeles area
    'Longitude': -118.3  # Los Angeles area
}


def create_feature_summary() -> str:
    """
    Create a formatted summary of all features
    
    Returns:
        Formatted string with feature information
    """
    summary = "CALIFORNIA HOUSING FEATURES\n"
    summary += "=" * 70 + "\n\n"
    
    for feature in FEATURE_DESCRIPTIONS.keys():
        summary += f"{feature} ({FEATURE_UNITS[feature]})\n"
        summary += f"  Description: {FEATURE_DESCRIPTIONS[feature]}\n"
        summary += f"  Typical value: {TYPICAL_VALUES[feature]}\n\n"
    
    return summary


if __name__ == "__main__":
    # Example usage and testing
    from config import FEATURE_RANGES
    
    print("InputValidator Module Test")
    print("=" * 70)
    
    # Create validator
    validator = InputValidator(FEATURE_RANGES)
    print(f"\n✓ Created validator with {len(FEATURE_RANGES)} features")
    
    # Test valid input
    print("\n--- Testing Valid Input ---")
    valid_input = {
        'MedInc': 3.8,
        'HouseAge': 28,
        'AveRooms': 5.4,
        'AveBedrms': 1.1,
        'Population': 3000,
        'AveOccup': 3.2,
        'Latitude': 34.2,
        'Longitude': -118.3
    }
    
    is_valid, errors = validator.validate_features(valid_input)
    print(f"Valid input: {is_valid}")
    if errors:
        print(f"Errors: {errors}")
    else:
        print("✓ No errors")
    
    # Test invalid input (out of range)
    print("\n--- Testing Invalid Input (Out of Range) ---")
    invalid_input = valid_input.copy()
    invalid_input['MedInc'] = 100  # Way too high
    
    is_valid, errors = validator.validate_features(invalid_input)
    print(f"Valid input: {is_valid}")
    print(f"Errors: {errors}")
    
    # Test missing feature
    print("\n--- Testing Missing Feature ---")
    incomplete_input = valid_input.copy()
    del incomplete_input['MedInc']
    
    is_valid, errors = validator.validate_features(incomplete_input)
    print(f"Valid input: {is_valid}")
    print(f"Errors: {errors}")
    
    # Show feature info
    print("\n--- Feature Information ---")
    feature_info = validator.get_feature_info()
    for feature, info in list(feature_info.items())[:2]:  # Show first 2
        print(f"{feature}:")
        print(f"  Range: {info['min']} to {info['max']}")
        print(f"  Description: {info['description'][:60]}...")
    
    print("\n" + "=" * 70)
    print("✓ All tests completed")
