Demo Video:
https://youtu.be/1WtwLWLUjhA


Inspiration
Having younger siblings who might need assistance with their work, and sometimes I’m just not able to make it on time; however, Marshmallow can take my place, and even though it can’t replace me, it’s the next best thing in their educational journey, leveraging cutting-edge AI tools without potentially harming them through unrestricted access.

What it does
Marshmallow is an AR buddy that assists primary children with their educational needs, adapting to their lingual diversity by speaking in their native language. Essentially, it analyzes the validity of the user’s work, points out any incorrect variables or numbers in the equations, congratulates when the answer is correct, and is willing to provide hints in case the user is struggling with the problem.

How we built it
It utilises Unity’s Augmented Reality Kit to create a virtual environment that is tracked to real world positions, which allows us to take a snapshot of the user’s work, and then utilize fine tuned models both through api inference lightweight models that provide lightning fast customized personal hints, it does this by recognising text using fine-tuned LLM’s, which is then converted to latex equations and parsed to verify the valdity of the equation, which is finally passed to these inference models to generate personalised feedback in a human voice. .

Challenges we ran into
We faced a specific challenge in determining the position mapping system, as there were inconsistencies between the scale used by Unity, the backend programs, and the differences caused by the dimensions of the user’s device. This led to incorrect mapping because each part of the program handled it differently. Another issue we encountered was that the latest stable build versions of Xcode and Unity were not compatible. The developers were working on a fix just this past week, which forced us to utilize AR alpha packages that were potentially less stable. This was because it was an emergency hot fix by the developers, and hence, we had to adapt to the circumstances.

Accomplishments that we're proud of
We are incredibly proud of our “almost” complete universal position mapping system, which unifies dimensions across Unity, various device types, and the backend script to create a uniform coordinate system. This allows the AR character to move to the exact location of the error and point to it. We also utilized various state machines to transition between dancing for a correct answer, pointing to a mistake in the equation, and walking towards the place of the error in the equation, with more animations pending to create a more lively AR buddy. .

What we learned
What we learned is that for every large project, you need to create thorough schemas and UML diagrams to get a clear, high-level overview of the various modular parts of the program that come together in different ways. This would have allowed us to work more efficiently, as although we did divide tasks, we had to backtrack quite a bit because of various dependencies we hadn’t considered, which extended our completion deadlines.

What's next for Marshmallow
We fine-tune the position mapping to ensure consistency across the backend scripts, Unity, and devices of different sizes, thereby ensuring universal functionality. Additionally, we would utilize custom-trained models, which would reduce wait times and costs, allowing us to scale this project more effectively in the future. We have implemented a multilingual speech model that enables users to learn in their native language, which helps them learn more optimally.

