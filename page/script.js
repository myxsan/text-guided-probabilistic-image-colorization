// Project page interactions: BibTeX copy button + reveal-on-scroll cards.

document.addEventListener("DOMContentLoaded", () => {
  // ── Copy BibTeX ──────────────────────────────────────────────────────
  const btn = document.getElementById("copy-bib");
  const bib = document.querySelector(".bib code");
  if (btn && bib) {
    btn.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(bib.textContent.trim());
        btn.textContent = "Copied ✓";
        btn.classList.add("copied");
        setTimeout(() => {
          btn.textContent = "Copy BibTeX";
          btn.classList.remove("copied");
        }, 1800);
      } catch (e) {
        btn.textContent = "Copy failed";
        setTimeout(() => (btn.textContent = "Copy BibTeX"), 1800);
      }
    });
  }

  // ── Reveal cards on scroll ───────────────────────────────────────────
  const cards = document.querySelectorAll(".card");
  cards.forEach((c) => {
    c.style.opacity = "0";
    c.style.transform = "translateY(14px)";
    c.style.transition = "opacity .55s ease, transform .55s ease";
  });
  const io = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.style.opacity = "1";
          entry.target.style.transform = "translateY(0)";
          io.unobserve(entry.target);
        }
      });
    },
    { rootMargin: "0px 0px -80px 0px", threshold: 0.05 }
  );
  cards.forEach((c) => io.observe(c));

  // ── Smooth active nav highlighting ───────────────────────────────────
  const navLinks = document.querySelectorAll(".pill-link[href^='#']");
  navLinks.forEach((link) => {
    link.addEventListener("click", (e) => {
      const id = link.getAttribute("href");
      if (id.startsWith("#")) {
        const target = document.querySelector(id);
        if (target) {
          e.preventDefault();
          target.scrollIntoView({ behavior: "smooth", block: "start" });
        }
      }
    });
  });
});
