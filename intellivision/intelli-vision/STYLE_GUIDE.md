# Intelli-Vision Frontend Style Guide

This style guide documents all custom utility classes defined in `src/index.css` and describes their purpose and intended use. Use these classes for consistent, unified styling across all components and pages.

---

## Primitives & Required Classes

The following React primitives are provided in `src/components/ui/Primitives.tsx` for consistent, maintainable UI:

### Card

- **Component:** `<Card>`
- **Required classes:** `glass-card`, `rounded-4xl`, `p-8`, `shadow-2xl`
- **Purpose:** Use for all card-like containers. Do not use raw divs with these classes directly.
- **Example:**
  ```jsx
  <Card>Content here</Card>
  ```

### Section

- **Component:** `<Section>`
- **Required class:** `content-card`
- **Purpose:** Use for major page or layout sections.
- **Example:**
  ```jsx
  <Section>Section content</Section>
  ```

### FeatureCard

- **Component:** `<FeatureCard>`
- **Required class:** `feature-card`
- **Purpose:** Use for feature highlights or list items.
- **Example:**
  ```jsx
  <FeatureCard>Feature content</FeatureCard>
  ```

### AppButton

- **Component:** `<AppButton color="primary|secondary|tertiary">`
- **Base classes:** `h-12`, `text-base`, `font-extrabold`, `button-hover`, `rounded-3xl`, `transition-all`, `px-6`
- **Color variants:**
  - `primary`: `bg-green-500/60`, `backdrop-blur-lg`, `text-white`, `shadow-lg`, `hover:bg-green-500/70`, `border`, `border-green-400`
  - `secondary`: `bg-cyan-500/60`, `backdrop-blur-lg`, `text-white`, `shadow-lg`, `hover:bg-cyan-500/70`, `border`, `border-cyan-400`
  - `tertiary`: `bg-white/5`, `hover:bg-white/15`, `text-white`, `border`, `border-white/20`, `shadow`, `backdrop-blur-xl`
- **Disabled state:** `disabled:opacity-50`, `disabled:cursor-not-allowed`
- **Purpose:** Use for all button actions. Do not use raw button elements with these classes directly.
- **Example:**
  ```jsx
  <AppButton color="primary">Save</AppButton>
  <AppButton color="secondary">Confirm</AppButton>
  <AppButton color="tertiary">Show Info</AppButton>
  ```

---

## Custom Utility Classes

### Backgrounds & Cards

- **.bg-navy-gradient**: Navy blue gradient background for hero sections or major panels.
- **.bg-navy-card**: Light card background with subtle gradient and blur for light mode.
- **.bg-navy-card-dark**: Card background for dark mode.
- **.bg-light-teal**: Light teal gradient background for highlights or sections.
- **.bg-navy-button**: Navy gradient for primary buttons.

### Text

- **.text-gradient**: Animated blue gradient text for headings or highlights.
- **.text-gradient-white**: Animated white/gray gradient text for light-on-dark emphasis.

### Borders & Shadows

- **.border-subtle**: Subtle shadow and border for cards or containers.

### Glassmorphism

- **.glass-card**: Glassy card effect with blur and border, for content cards.
- **.glass-navbar**: Glassy effect for navigation bars.
- **.glass-intense**: Stronger glass effect for overlays or modals.
- **.glass-subtle**: Subtle glass effect for backgrounds.
- **.glass-button**: Glassy button style for secondary actions.

### Animations & Effects

- **.hero-glow**: Adds a glowing effect to hero sections.
- **.floating**: Slow floating animation for decorative elements.
- **.bounce-in**: Bounce-in entrance animation.
- **.slide-up**: Slide-up entrance animation.
- **.slide-in-left**: Slide-in from left animation.
- **.slide-in-right**: Slide-in from right animation.
- **.fade-in-up**: Fade-in and move up animation.
- **.scale-in**: Scale-in entrance animation.
- **.rotate-in**: Rotational entrance animation.
- **.stagger-fade**: Staggered fade-in for lists (use nth-child for delay).
- **.micro-bounce**: Small bounce on active/press.
- **.glow-hover**: Glow effect on hover.

### Cards & Layout

- **.card-hover**: Card hover effect with lift and shadow.
- **.card-base**: Base card style with padding, border, blur, and shadow.
- **.content-card**: Enhanced card for landing page sections (uses glass and shadow).
- **.feature-card**: Feature list item card (light glass, rounded, border).

### Buttons

- **.button-hover**: Button hover effect with lift and shadow.
- **.glass-button**: Glassy button style for secondary actions.

### Rounding

- **.rounded-3xl**: 1.5rem border radius.
- **.rounded-4xl**: 2rem border radius.
- **.rounded-5xl**: 2.5rem border radius.

### Page & Route Transitions

- **.page-enter, .page-enter-active, .page-exit, .page-exit-active**: Page transition animations.
- **.route-enter, .route-enter-active, .route-exit, .route-exit-active**: Route transition animations.

---

## Usage Guidelines

- **Always use these classes and primitives for backgrounds, cards, buttons, and effects instead of ad-hoc Tailwind or inline styles.**
- **For colors, backgrounds, and borders, use the CSS variables defined in `:root` and `.dark` in `index.css` (e.g., `bg-[hsl(var(--background))]`).**
- **If you need a new utility, add it to `index.css` and document it here.**
- **For dark mode, ensure your component respects the `.dark` class and uses variables.**

---

## Example

```jsx
<Card>
  <h2 className="text-gradient text-3xl mb-4">Welcome!</h2>
  <p className="text-lg">
    This card uses unified styles from the design system.
  </p>
</Card>

<AppButton color="primary">Save</AppButton>
```

---

Keep this guide up to date as you add or modify custom utilities and primitives.
