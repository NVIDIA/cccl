📝 `d.cu:4`: `error: this is a very long error message that should be truncated...`

📍 Location: `d.cu:4`

🔍 Full Error:

<pre>
d.cu(4): error: this is a very long error message that should be truncated to ensure that the summary does not exceed the limit and still has balanced backticks
a.cu(3): error: expected a ";"
  }
  ^

1 error detected in the compilation of "a.cu".
</pre>

📝 `a.cu:3`: `error: expected a ";"`

📍 Location: `a.cu:3`

🔍 Full Error:

<pre>
a.cu(3): error: expected a ";"
  }
  ^

1 error detected in the compilation of "a.cu".
</pre>

📝 `b.cu:3`: `error: expected a ";"`

📍 Location: `b.cu:3`

🔍 Full Error:

<pre>
b.cu(3): error: expected a ";"
  }
  ^

1 error detected in the compilation of "b.cu".
</pre>

📝 `c.cu:2`: `error: identifier "y" is undefined`

📍 Location: `c.cu:2`

🔍 Full Error:

<pre>
c.cu(2): error: identifier "y" is undefined
      int x = y;
              ^

1 error detected in the compilation of "c.cu".
</pre>
