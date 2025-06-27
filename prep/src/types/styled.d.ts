import 'styled-components';

declare module 'styled-components' {
  export interface DefaultTheme {
    colors: {
      primary: string;
      secondary: string;
      accent: string;
      purple: string;
      background: string;
      surface: string;
      surfaceLight: string;
      text: string;
      textSecondary: string;
      border: string;
      success: string;
      warning: string;
      error: string;
    };
    fonts: {
      main: string;
      mono: string;
      math: string;
    };
    breakpoints: {
      mobile: string;
      tablet: string;
      desktop: string;
      wide: string;
    };
    spacing: {
      xs: string;
      sm: string;
      md: string;
      lg: string;
      xl: string;
      xxl: string;
    };
    radii: {
      sm: string;
      md: string;
      lg: string;
      xl: string;
    };
    shadows: {
      sm: string;
      md: string;
      lg: string;
      xl: string;
    };
  }
}
